"""Mapa y localizacion propios, en vivo y sin mapa previo (SLAM 2D): el origen del mapa es donde arranca el robot.
Rumbo: giroscopio (con la correccion opcional del LiDAR). Posicion: cada barrido se calza contra los puntos de pared ya vistos.
Mapa: grilla de ocupacion que se actualiza con cada barrido. La pose real solo se usa para medir el error."""
import cv2
import numpy as np

import localizacion
from mapa_ocupacion import L_LIBRE, L_OCC, L_TOPE, RES, UMBRAL_OCC, dibujar
from odometria_lidar import calzar, puntos_2d, un_punto_por_celda

TAM_MAPA = 40.0       # m de lado de la grilla, centrada en el punto de partida
ACTUALIZAR_1_DE = 2   # el mapa y los puntos de pared se actualizan con 1 de cada N barridos (5 Hz); la posicion se calza en todos


def slam_crear(lidar_offset_xy, sesgo_grados_s=0.0, ruido_rad_s=0.0, corregir_rumbo=False):
    """lidar_offset_xy: donde esta el LiDAR respecto del centro del robot (x, y en el marco del torso).
    El origen del mapa queda en el centro del robot al arrancar, con el rumbo con el que arranca."""
    n = int(TAM_MAPA / RES)
    return {"L": np.zeros((n, n), dtype=np.float32), "lo": np.full(2, -TAM_MAPA / 2), "pared": np.zeros((0, 2)),
            "R": np.eye(3), "tr": np.asarray(lidar_offset_xy, dtype=float).copy(), "n": 0, "real0": None,
            "sesgo": np.deg2rad(sesgo_grados_s), "ruido": ruido_rad_s, "corregir": corregir_rumbo,
            "rng": np.random.default_rng(0), "est": [], "real": [], "err": [], "err_yaw": []}


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
    obst, piso = puntos_2d(ranges, dirs, slam["R"], rango_max)
    src = un_punto_por_celda(obst)
    if slam["n"] > 0:
        th, tr, ok = calzar(src, slam["pared"], 0.0, slam["tr"], corregir_rumbo=slam["corregir"])
        if ok:                                          # con muy pocos puntos se queda donde estaba
            slam["tr"] = tr
            if slam["corregir"]:
                slam["R"] = localizacion.girar_z(slam["R"], th)
    if slam["n"] % ACTUALIZAR_1_DE == 0:
        mapa_actualizar(slam, slam["tr"], obst + slam["tr"], piso + slam["tr"])
        if not slam.get("congelado"):   # congelado: se sigue completando la grilla, pero la posicion se calza siempre contra la referencia de la exploracion
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


def slam_resumen(slam):
    e = np.array(slam["err"])
    L = slam["L"]
    return (f"error de posicion {e[-1]:.3f} m (medio {e.mean():.3f}, maximo {e.max():.3f}), rumbo {slam['err_yaw'][-1]:.2f} grados; "
            f"mapa: {int((L > UMBRAL_OCC).sum())} celdas ocupadas, {int((L < -0.2).sum())} libres")


def slam_recortar(slam, margen=10):
    """El mapa sin el borde sin tocar: (grilla, origen en metros)."""
    f, c = np.nonzero(slam["L"] != 0)
    f0, f1, c0, c1 = max(f.min() - margen, 0), f.max() + margen, max(c.min() - margen, 0), c.max() + margen
    return slam["L"][f0:f1 + 1, c0:c1 + 1].copy(), slam["lo"] + np.array([c0, f0]) * RES


def slam_guardar(slam, base, escenario, waypoints):
    """Guarda el mapa (npz + png). Celeste = recorrido estimado, verde = real, rojo = waypoints."""
    L, lo = slam_recortar(slam)
    np.savez_compressed(base + ".npz", logodds=L, origen_xy=lo, resolucion=RES, escenario=escenario)
    dibujar(L, lo, np.array(slam["est"]), np.asarray(waypoints, dtype=float), base + ".png", trayecto_ref=np.array(slam["real"]))
