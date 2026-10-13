"""Localizacion en vivo: el robot estima donde esta comparando cada barrido del LiDAR contra el mapa guardado
(giroscopio para el rumbo, LiDAR contra el mapa para la posicion). La posicion real solo se usa para medir el error."""
import numpy as np
from scipy.spatial.transform import Rotation

from mapa_ocupacion import RES, UMBRAL_OCC
from odometria_lidar import calzar, puntos_2d, un_punto_por_celda


def loc_crear(L, lo):
    """Estado inicial: el robot parte donde nacio la grabacion del mapa (origen del mapa, sin rotar)."""
    fila, col = np.nonzero(L > UMBRAL_OCC)
    mapa = lo + (np.c_[col, fila] + 0.5) * RES     # centros de las celdas ocupadas, en el marco del mapa
    return {"mapa": mapa, "R": np.eye(3), "tr": np.zeros(2), "yaw_real0": None,
            "est": [], "real": [], "err": [], "err_yaw": []}


def loc_girar(loc, gyro, dt):
    """Suma al rumbo estimado lo que giro el torso en este paso (giroscopio)."""
    loc["R"] = loc["R"] @ Rotation.from_rotvec(np.asarray(gyro, dtype=float) * dt).as_matrix()


def loc_barrido(loc, dirs, ranges, rango_max, real_xy, R_real):
    """Calza un barrido contra el mapa y guarda el error contra la pose real (real_xy en el marco del mapa, R_real del LiDAR)."""
    obst, _ = puntos_2d(ranges, dirs, loc["R"], rango_max)
    _, tr, ok = calzar(un_punto_por_celda(obst), loc["mapa"], 0.0, loc["tr"])
    if ok:                                          # con muy pocos puntos se queda donde estaba
        loc["tr"] = tr
    yaw_real = np.arctan2(R_real[1, 0], R_real[0, 0])
    if loc["yaw_real0"] is None:
        loc["yaw_real0"] = yaw_real
    dy = np.arctan2(loc["R"][1, 0], loc["R"][0, 0]) - (yaw_real - loc["yaw_real0"])
    loc["est"].append(loc["tr"].copy())
    loc["real"].append(np.asarray(real_xy, dtype=float).copy())
    loc["err"].append(float(np.linalg.norm(loc["tr"] - real_xy)))
    loc["err_yaw"].append(abs(np.degrees(np.arctan2(np.sin(dy), np.cos(dy)))))


def loc_resumen(loc):
    e = np.array(loc["err"])
    return f"error de posicion {e[-1]:.3f} m (medio {e.mean():.3f}, maximo {e.max():.3f}), rumbo {loc['err_yaw'][-1]:.2f} grados"


def loc_pose_base(loc, off, lidar_offset):
    """Pose estimada del robot en el mundo: posicion (x, y, 0) y cuaternion (w, x, y, z) que solo lleva el rumbo.
    off: desplazamiento mundo -> marco del mapa; lidar_offset: posicion del LiDAR en el marco del torso."""
    R = loc["R"]
    xy = loc["tr"] + off - (R @ lidar_offset)[:2]      # del LiDAR (lo que se estima) al centro del robot
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([xy[0], xy[1], 0.0]), np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
