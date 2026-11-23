"""Mapa de obstaculos en vivo: acumula lo que ven el LiDAR y la camara de profundidad mientras el robot camina,
alineado con la grilla del mapa guardado (mismas celdas), para detectar lo que el mapa viejo no tiene."""
import cv2
import numpy as np

import planificador
from mapa_ocupacion import RES

ALTURA_OBST = (0.10, 1.50)   # m sobre el piso: mas bajo es piso, mas alto es techo
MIN_VISTAS = 2               # una celda cuenta como obstaculo si se la vio en al menos 2 barridos/imagenes


def vivo_crear(L):
    return np.zeros(L.shape, dtype=np.int16)


def puntos_lidar(data, body_id, dirs, origen, ranges, rango_max):
    """Puntos (N,3) del mundo donde pego cada rayo del LiDAR."""
    R = data.xmat[body_id].reshape(3, 3)
    ok = np.isfinite(ranges) & (ranges <= rango_max)
    return origen + (dirs[ok] @ R.T) * ranges[ok, None]


def puntos_profundidad(data, cam_id, depth, f, paso=4):
    """Puntos (N,3) del mundo a partir de la imagen de profundidad (camara MuJoCo: x derecha, y arriba, mira por -z)."""
    alto, ancho = depth.shape
    v, u = np.mgrid[0:alto:paso, 0:ancho:paso]
    d = depth[::paso, ::paso]
    ok = np.isfinite(d)
    d, u, v = d[ok], u[ok], v[ok]
    pc = np.stack([(u - ancho / 2) / f * d, -(v - alto / 2) / f * d, -d], axis=1)
    return data.cam_xpos[cam_id] + pc @ data.cam_xmat[cam_id].reshape(3, 3).T


def vivo_agregar(vivo, lo, off, pts):
    """Suma 1 a cada celda que tiene algun punto con altura de obstaculo (una vez por barrido/imagen).
    lo: origen del mapa; off: desplazamiento mundo -> marco del mapa."""
    p = pts[(pts[:, 2] > ALTURA_OBST[0]) & (pts[:, 2] < ALTURA_OBST[1])]
    c = np.floor((p[:, :2] - off - lo) / RES).astype(int)
    ok = (c[:, 0] >= 0) & (c[:, 0] < vivo.shape[1]) & (c[:, 1] >= 0) & (c[:, 1] < vivo.shape[0])
    c = np.unique(c[ok], axis=0)
    vivo[c[:, 1], c[:, 0]] += 1


def obstaculos_nuevos(vivo, L):
    """Celdas que el mapa guardado daba por LIBRES y ahora se ven ocupadas (varias veces, y a mas de 15 cm de un obstaculo conocido).
    Las celdas desconocidas no cuentan: el mapa viejo nunca las observo, asi que no se contradice con nada."""
    cerca_de_obstaculo = cv2.dilate((L > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return (vivo >= MIN_VISTAS) & (L < -0.2) & ~cerca_de_obstaculo


def replanificar(L, lo, off, nuevas, pos, waypoints, idx, destino):
    """Si lo que falta de la ruta cruza obstaculos nuevos, calcula otra ruta desde 'pos' (x, y del mundo).
    Devuelve (waypoints del mundo, mensaje). waypoints None y mensaje '' = la ruta sigue libre, no hace falta nada;
    waypoints None y mensaje con texto = la ruta esta bloqueada y no hay otra."""
    L2 = L.copy()
    L2[nuevas] = 4.0                                        # los obstaculos nuevos pasan a ser parte del mapa
    trans_viejo, _ = planificador.zona_transitable(L)
    trans_nuevo, _ = planificador.zona_transitable(L2)
    puntos = [np.asarray(pos, dtype=float) - off] + [np.asarray(w, dtype=float) - off for w in waypoints[idx:]]
    celdas = [planificador.a_celda(p, lo) for p in puntos]
    bloqueada = any(planificador.linea_libre(trans_viejo, a, b) and not planificador.linea_libre(trans_nuevo, a, b)
                    for a, b in zip(celdas[:-1], celdas[1:]))
    if not bloqueada:
        return None, ""
    wp, _, msg = planificador.planificar(L2, lo, puntos[0], np.asarray(destino, dtype=float) - off)
    if wp is None:
        return None, msg
    return (wp + off).tolist(), msg
