"""Estima el recorrido del robot SIN usar su posicion real: giroscopio (rumbo) + LiDAR (desplazamiento).
La posicion real solo se usa al final, para medir el error.
Uso:  python navegacion/odometria_lidar.py [grabaciones/corrida_X.npz]   (sin argumento usa la ultima)"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mapa_ocupacion import construir_mapa, direcciones_lidar, dibujar, RES, UMBRAL_OCC

BANDA_LIDAR = (-0.83, 0.57)   # alturas (m) respecto al LiDAR que cuentan como obstaculo (piso mas abajo, techo mas arriba)
CELDA_PUNTOS = 0.10           # se deja 1 punto por celda de 10 cm para que el calce sea rapido
DMAX_INICIAL = 0.5            # distancia maxima (m) para emparejar puntos en la primera vuelta del calce
CORREGIR_RUMBO = False        # False: el rumbo lo da solo el giroscopio y el LiDAR estima solo el desplazamiento


def orientacion_giroscopio(t, gyro):
    """Orientacion del torso (matriz 3x3 por paso) integrando el giroscopio, partiendo de 'sin rotar'."""
    R = [np.eye(3)]
    for k in range(len(t) - 1):
        R.append(R[-1] @ Rotation.from_rotvec(gyro[k] * (t[k + 1] - t[k])).as_matrix())
    return np.array(R)


def puntos_2d(r, dirs, R, rango_max):
    """Puntos del barrido alrededor del LiDAR (x,y en el plano horizontal), separados en obstaculo / piso."""
    ok = np.isfinite(r) & (r <= rango_max)
    q = (dirs[ok] @ R.T) * r[ok, None]
    obst = q[(q[:, 2] > BANDA_LIDAR[0]) & (q[:, 2] < BANDA_LIDAR[1])][:, :2]
    piso = q[q[:, 2] <= BANDA_LIDAR[0]][:, :2]
    return obst, piso


def un_punto_por_celda(p):
    _, u = np.unique(np.floor(p / CELDA_PUNTOS).astype(np.int64), axis=0, return_index=True)
    return p[u]


def calzar(src, mapa, th, tr, iters=20, dmax=DMAX_INICIAL, corregir_rumbo=None):
    """ICP 2D: busca el giro 'th' y el desplazamiento 'tr' que acercan los puntos 'src' al 'mapa'."""
    arbol = cKDTree(mapa)
    if corregir_rumbo is None:
        corregir_rumbo = CORREGIR_RUMBO
    for _ in range(iters):
        c, s = np.cos(th), np.sin(th)
        p = src @ np.array([[c, -s], [s, c]]).T + tr
        dist, idx = arbol.query(p, distance_upper_bound=dmax)
        m = np.isfinite(dist)
        if m.sum() < 30:
            return th, tr, False
        a, b = p[m], mapa[idx[m]]
        ca, cb = a.mean(0), b.mean(0)
        if corregir_rumbo:
            U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
            Rd = Vt.T @ U.T
            if np.linalg.det(Rd) < 0:
                Vt[1] *= -1
                Rd = Vt.T @ U.T
        else:
            Rd = np.eye(2)
        td = cb - Rd @ ca
        dth = np.arctan2(Rd[1, 0], Rd[0, 0])
        th, tr = th + dth, Rd @ tr + td
        dmax = max(0.15, dmax * 0.8)
        if abs(dth) < 1e-5 and np.linalg.norm(td) < 1e-4:
            break
    return th, tr, True


def estimar_recorrido(d, cal):
    """Devuelve la pose estimada del LiDAR por barrido (giro corregido, x, y) y los puntos de cada barrido ya ubicados."""
    dirs = direcciones_lidar(cal)
    rmax = cal["lidar"]["rango_m"][1]
    Rg = orientacion_giroscopio(d["t"], d["imu"][:, 0:3].astype(float))
    est, barridos, mapa = [], [], np.zeros((0, 2))
    for i, k in enumerate(d["lidar_idx"]):
        obst, piso = puntos_2d(d["lidar"][i], dirs, Rg[k], rmax)
        src = un_punto_por_celda(obst)
        if i == 0:
            th, tr = 0.0, np.zeros(2)
        else:
            th, tr, ok = calzar(src, mapa, est[-1][0], est[-1][1:3])
            if not ok:   # muy pocos puntos: se queda donde estaba
                th, tr = est[-1][0], est[-1][1:3]
        est.append(np.r_[th, tr])
        c, s = np.cos(th), np.sin(th)
        M = np.array([[c, -s], [s, c]]).T
        pu, pp = src @ M + tr, piso @ M + tr
        if i % 2 == 0:   # el mapa guarda un punto por celda, asi no crece sin limite
            mapa = un_punto_por_celda(np.vstack([mapa, pu]))
        barridos.append((np.r_[tr, 0.0], np.c_[obst @ M + tr, np.zeros(len(obst))], np.c_[pp, np.zeros(len(pp))]))
    yaw_giro = np.array([np.arctan2(Rg[k][1, 0], Rg[k][0, 0]) for k in d["lidar_idx"]])
    return np.array(est), np.unwrap(yaw_giro), barridos


def recorrido_real(d, cal):
    """Verdad terreno del LiDAR en el marco inicial (para comparar, no para estimar)."""
    k = d["lidar_idx"]
    off = np.array(cal["lidar"]["pos_en_base_m"])
    R0 = d["torso_mat"][0]
    yaw0 = np.arctan2(R0[1, 0], R0[0, 0])
    c, s = np.cos(yaw0), np.sin(yaw0)
    P = np.array([d["torso_pos"][j] + d["torso_mat"][j] @ off for j in k])[:, :2]
    yaw = np.unwrap([np.arctan2(d["torso_mat"][j][1, 0], d["torso_mat"][j][0, 0]) for j in k])
    return (P - P[0]) @ np.array([[c, s], [-s, c]]).T, yaw - yaw[0], (yaw0, d["torso_pos"][0, :2])


def main():
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(raiz, "grabaciones", "*.npz")))[-1]
    d = np.load(ruta)
    d = {k: d[k] for k in d.files}
    with open(os.path.join(raiz, "sensores.json")) as f:
        cal = json.load(f)
    est, yaw_giro, barridos = estimar_recorrido(d, cal)
    real, yaw_real, (yaw0, xy0) = recorrido_real(d, cal)

    e = np.linalg.norm(est[:, 1:3] - real, axis=1)
    camino = np.linalg.norm(np.diff(real, axis=0), axis=1).sum()
    e_yaw = np.rad2deg(np.abs((yaw_giro + est[:, 0]) - yaw_real))
    print(f"[odometria] recorrido real: {camino:.1f} m en {d['t'][d['lidar_idx'][-1]]:.0f} s")
    print(f"[odometria] error de posicion: medio {e.mean():.3f} m, maximo {e.max():.3f} m, al final {e[-1]:.3f} m ({e[-1] / camino * 100:.2f}% del recorrido)")
    print(f"[odometria] error de rumbo: medio {e_yaw.mean():.2f} grados, maximo {e_yaw.max():.2f} grados")

    carpeta = os.path.join(raiz, "mapas")
    os.makedirs(carpeta, exist_ok=True)
    base = os.path.join(carpeta, os.path.splitext(os.path.basename(ruta))[0] + "_mapa_estimado")
    np.savez_compressed(base.replace("_mapa_estimado", "_poses_estimadas") + ".npz",
                        t=d["t"][d["lidar_idx"]], x=est[:, 1], y=est[:, 2], yaw=yaw_giro + est[:, 0])
    L, lo, _ = construir_mapa([(o, occ, pis) for o, occ, pis in barridos[::2]], np.zeros(2), 1.0, 0.0)
    np.savez_compressed(base + ".npz", logodds=L, origen_xy=lo, resolucion=RES, escenario=d["escenario"])   # la grilla, para el planificador
    c, s = np.cos(yaw0), np.sin(yaw0)
    wp = (d["waypoints"] - xy0) @ np.array([[c, s], [-s, c]]).T
    dibujar(L, lo, est[:, 1:3], wp, base + ".png", trayecto_ref=real)
    print(f"[odometria] mapa con posicion estimada: {base}.png (celeste = estimado, verde = real)")


if __name__ == "__main__":
    main()
