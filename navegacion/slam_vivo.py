"""Mapa y localizacion propios, en vivo y sin mapa previo (SLAM 2D): el origen del mapa es donde arranca el robot.
Rumbo: giroscopio (con la correccion opcional del LiDAR). Posicion: cada barrido se calza contra los puntos de pared ya vistos.
Mapa: grilla de ocupacion que se actualiza con cada barrido. La pose real solo se usa para medir el error."""
import cv2
import numpy as np

import localizacion
from mapa_ocupacion import L_LIBRE, L_OCC, L_TOPE, RES, UMBRAL_OCC, dibujar
from odometria_lidar import calzar, puntos_2d, un_punto_por_celda

TAM_MAPA = 40.0       # m de lado de la grilla, centrada en el punto de partida
RANGO_UTIL = 10.0     # m: mas lejos que esto no se usa ni para calzar ni para el mapa (el error angular se agranda con la distancia y en espacios abiertos se marcaria como libre media ciudad)
BAJOS_Z = (0.10, 1.30)    # m sobre el piso: lo que la camara de profundidad ve en esta franja se cuenta como obstaculo bajo (el LiDAR lo ve mal o no lo ve)
BAJOS_DIST = (0.4, 3.0)   # m: solo se usan puntos de profundidad en este rango (mas lejos la pose estimada los corre demasiado)
BAJOS_VISTAS = 2          # una celda cuenta como obstaculo bajo si se la vio en al menos tantas imagenes
TILT_GRAVEDAD = True      # la inclinacion (roll, pitch) sale de la gravedad como en un IMU real, no de integrar el giroscopio (que se corre unos grados en minutos); el rumbo sigue siendo el estimado
ACTUALIZAR_1_DE = 2   # el mapa y los puntos de pared se actualizan con 1 de cada N barridos (5 Hz); la posicion se calza en todos


def slam_crear(lidar_offset_xy, sesgo_grados_s=0.0, ruido_rad_s=0.0, corregir_rumbo=False):
    """lidar_offset_xy: donde esta el LiDAR respecto del centro del robot (x, y en el marco del torso).
    El origen del mapa queda en el centro del robot al arrancar, con el rumbo con el que arranca."""
    n = int(TAM_MAPA / RES)
    return {"L": np.zeros((n, n), dtype=np.float32), "lo": np.full(2, -TAM_MAPA / 2), "pared": np.zeros((0, 2)),
            "R": np.eye(3), "tr": np.asarray(lidar_offset_xy, dtype=float).copy(), "n": 0, "real0": None,
            "sesgo": np.deg2rad(sesgo_grados_s), "ruido": ruido_rad_s, "corregir": corregir_rumbo,
            "rng": np.random.default_rng(0), "est": [], "real": [], "err": [], "err_yaw": [],
            "bajos": np.zeros((n, n), dtype=np.uint8)}


def celdas(slam, p):
    return np.floor((p - slam["lo"]) / RES).astype(np.int32)


def mapa_actualizar(slam, origen, occ, piso):
    """Suma al mapa lo que vio un barrido: libre a lo largo de cada rayo, ocupado donde pego."""
    L = slam["L"]
    alto, ancho = L.shape
    ce = celdas(slam, occ)
    ce = ce[(ce[:, 0] >= 0) & (ce[:, 0] < ancho) & (ce[:, 1] >= 0) & (ce[:, 1] < alto)]
    extremos = np.vstack([ce, celdas(slam, piso)])
    segs = np.empty((len(extremos), 2, 1, 2), dtype=np.int32)
    segs[:, 0, 0, :] = celdas(slam, origen)
    segs[:, 1, 0, :] = extremos
    libre = np.zeros(L.shape, dtype=np.uint8)
    cv2.polylines(libre, list(segs), False, 1)
    ocup = np.zeros(L.shape, dtype=bool)
    ocup[ce[:, 1], ce[:, 0]] = True
    L[(libre > 0) & ~ocup] += L_LIBRE
    L[ocup] += L_OCC
    np.clip(L, -L_TOPE, L_TOPE, out=L)


def slam_barrido(slam, dirs, ranges, rango_max, lidar_origin, R_real, torso_xy):
    """Un barrido: calza contra lo ya visto (posicion), actualiza el mapa y guarda el error contra la pose real.
    lidar_origin, R_real, torso_xy: pose real del LiDAR y del torso (solo para medir el error)."""
    if TILT_GRAVEDAD:
        d_yaw = np.arctan2(slam["R"][1, 0], slam["R"][0, 0]) - np.arctan2(R_real[1, 0], R_real[0, 0])
        c, s = np.cos(d_yaw), np.sin(d_yaw)
        slam["R"] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.asarray(R_real, dtype=float).reshape(3, 3)   # tilt real, rumbo estimado
    obst, piso = puntos_2d(ranges, dirs, slam["R"], min(rango_max, RANGO_UTIL))
    src = un_punto_por_celda(obst)
    if slam["n"] > 0:
        th, tr, ok = calzar(src, slam["pared"], 0.0, slam["tr"], corregir_rumbo=slam["corregir"])
        if ok:                                          # con muy pocos puntos se queda donde estaba
            slam["tr"] = tr
            if slam["corregir"]:
                slam["R"] = localizacion.girar_z(slam["R"], th)
    if slam["n"] % ACTUALIZAR_1_DE == 0 and not slam.get("congelado"):   # congelado (al empezar la inspeccion): ni el mapa ni la referencia cambian, la posicion se calza contra lo que dejo la exploracion
        mapa_actualizar(slam, slam["tr"], obst + slam["tr"], piso + slam["tr"])
        slam["pared"] = un_punto_por_celda(np.vstack([slam["pared"], src + slam["tr"]]))
    slam["n"] += 1
    # error contra la pose real, en el marco del mapa (origen en el torso al arrancar, con el rumbo con el que arranco)
    yaw_real = np.arctan2(R_real[1, 0], R_real[0, 0])
    if slam["real0"] is None:
        slam["real0"] = (np.asarray(torso_xy, dtype=float).copy(), yaw_real)
    xy0, yaw0 = slam["real0"]
    c, s = np.cos(yaw0), np.sin(yaw0)
    real_xy = np.array([[c, s], [-s, c]]) @ (np.asarray(lidar_origin[:2], dtype=float) - xy0)
    dy = np.arctan2(slam["R"][1, 0], slam["R"][0, 0]) - (yaw_real - yaw0)
    slam["est"].append(slam["tr"].copy())
    slam["real"].append(real_xy)
    slam["err"].append(float(np.linalg.norm(slam["tr"] - real_xy)))
    slam["err_yaw"].append(abs(np.degrees(np.arctan2(np.sin(dy), np.cos(dy)))))


def slam_R_camara(slam, data, cam_id, torso_id):
    """Orientacion de la camara en el marco del mapa: la inclinacion (roll y pitch) es la que mide la gravedad (IMU del robot, la integrada con el
    giroscopio se corre unos grados en minutos), el rumbo es el estimado por el SLAM. El rumbo real se resta y no entra."""
    Rt = data.xmat[torso_id].reshape(3, 3)
    d = np.arctan2(slam["R"][1, 0], slam["R"][0, 0]) - np.arctan2(Rt[1, 0], Rt[0, 0])
    c, s = np.cos(d), np.sin(d)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ data.cam_xmat[cam_id].reshape(3, 3)


def slam_bajos(slam, data, cam_id, torso_id, depth, f, paso=2):
    """Suma a la capa de obstaculos bajos lo que ve la camara de profundidad entre BAJOS_Z de altura. Se usa solo para planificar:
    no entra al calce del LiDAR ni al mapa de paredes. La orientacion es la estimada (roll y pitch del giroscopio), la altura de la camara es la del robot."""
    alto, ancho = depth.shape
    v, u = np.mgrid[0:alto:paso, 0:ancho:paso]
    d = depth[::paso, ::paso]
    ok = np.isfinite(d) & (d > BAJOS_DIST[0]) & (d < BAJOS_DIST[1])
    d, u, v = d[ok], u[ok], v[ok]
    if len(d) == 0:
        return
    pc = np.stack([(u - ancho / 2) / f * d, -(v - alto / 2) / f * d, -d], axis=1)
    pm = pc @ slam_R_camara(slam, data, cam_id, torso_id).T
    z = data.cam_xpos[cam_id][2] + pm[:, 2]
    bajo = (z > BAJOS_Z[0]) & (z < BAJOS_Z[1])
    if not bajo.any():
        return
    ce = celdas(slam, slam["tr"] + pm[bajo, :2])
    B = slam["bajos"]
    ce = ce[(ce[:, 0] >= 0) & (ce[:, 0] < B.shape[1]) & (ce[:, 1] >= 0) & (ce[:, 1] < B.shape[0])]
    ce = np.unique(ce, axis=0)
    B[ce[:, 1], ce[:, 0]] = np.minimum(B[ce[:, 1], ce[:, 0]] + 1, 250)


def slam_resumen(slam):
    e = np.array(slam["err"])
    L = slam["L"]
    return (f"error de posicion {e[-1]:.3f} m (medio {e.mean():.3f}, maximo {e.max():.3f}), rumbo {slam['err_yaw'][-1]:.2f} grados; "
            f"mapa: {int((L > UMBRAL_OCC).sum())} celdas ocupadas, {int((L < -0.2).sum())} libres")


def slam_recortar(slam, margen=10, bajos=False):
    """El mapa sin el borde sin tocar: (grilla, origen en metros). Con bajos=True, los obstaculos bajos vistos por la camara
    figuran como ocupados (salvo pegados a una pared, donde son la propia pared mal ubicada); para planificar, no para medir paredes."""
    f, c = np.nonzero(slam["L"] != 0)
    f0, f1, c0, c1 = max(f.min() - margen, 0), f.max() + margen, max(c.min() - margen, 0), c.max() + margen
    L = slam["L"][f0:f1 + 1, c0:c1 + 1].copy()
    if bajos:
        B = slam["bajos"][f0:f1 + 1, c0:c1 + 1] >= BAJOS_VISTAS
        pared = cv2.dilate((L > UMBRAL_OCC).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        L[B & ~pared] = L_TOPE
    return L, slam["lo"] + np.array([c0, f0]) * RES


def slam_guardar(slam, base, escenario, waypoints):
    """Guarda el mapa (npz + png). Celeste = recorrido estimado, verde = real, rojo = waypoints."""
    L, lo = slam_recortar(slam, bajos=True)
    np.savez_compressed(base + ".npz", logodds=L, origen_xy=lo, resolucion=RES, escenario=escenario)
    dibujar(L, lo, np.array(slam["est"]), np.asarray(waypoints, dtype=float), base + ".png", trayecto_ref=np.array(slam["real"]))
