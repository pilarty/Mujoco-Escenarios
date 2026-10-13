"""Localizacion en vivo: el robot estima donde esta comparando cada barrido del LiDAR contra el mapa guardado
(giroscopio para el rumbo, LiDAR contra el mapa para la posicion). La posicion real solo se usa para medir el error."""
import numpy as np
from scipy.spatial.transform import Rotation

from mapa_ocupacion import RES, UMBRAL_OCC
from odometria_lidar import calzar, puntos_2d, un_punto_por_celda


def loc_crear(L, lo, sesgo_grados_s=0.0, ruido_rad_s=0.0, corregir_rumbo=False):
    """Estado inicial: el robot parte donde nacio la grabacion del mapa (origen del mapa, sin rotar).
    sesgo_grados_s / ruido_rad_s: error que se le agrega al rumbo del giroscopio para simular uno real
    (el del simulador es perfecto). corregir_rumbo: el LiDAR corrige el rumbo contra el mapa en cada barrido."""
    fila, col = np.nonzero(L > UMBRAL_OCC)
    mapa = lo + (np.c_[col, fila] + 0.5) * RES     # centros de las celdas ocupadas, en el marco del mapa
    return {"mapa": mapa, "R": np.eye(3), "tr": np.zeros(2), "yaw_real0": None,
            "sesgo": np.deg2rad(sesgo_grados_s), "ruido": ruido_rad_s, "corregir": corregir_rumbo,
            "rng": np.random.default_rng(0), "est": [], "real": [], "err": [], "err_yaw": []}


def girar_z(R, ang):
    """Gira la orientacion R 'ang' radianes alrededor de la vertical del mapa."""
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R


def loc_girar(loc, gyro, dt):
    """Suma al rumbo estimado lo que giro el torso en este paso (giroscopio). Roll y pitch quedan bien (en un robot real
    los corrige la gravedad); al rumbo se le suma el sesgo y el ruido simulados, que es lo que de verdad deriva."""
    loc["R"] = loc["R"] @ Rotation.from_rotvec(np.asarray(gyro, dtype=float) * dt).as_matrix()
    loc["R"] = girar_z(loc["R"], (loc["sesgo"] + loc["ruido"] * loc["rng"].standard_normal()) * dt)


def loc_barrido(loc, dirs, ranges, rango_max, real_xy, R_real):
    """Calza un barrido contra el mapa y guarda el error contra la pose real (real_xy en el marco del mapa, R_real del LiDAR)."""
    obst, _ = puntos_2d(ranges, dirs, loc["R"], rango_max)
    th, tr, ok = calzar(un_punto_por_celda(obst), loc["mapa"], 0.0, loc["tr"], corregir_rumbo=loc["corregir"])
    if ok:                                          # con muy pocos puntos se queda donde estaba
        loc["tr"] = tr
        if loc["corregir"]:                         # el giro que pidio el calce se le suma al rumbo del giroscopio
            loc["R"] = girar_z(loc["R"], th)
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
