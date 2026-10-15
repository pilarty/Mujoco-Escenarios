"""Inspeccion de paredes: despues de explorar, el robot va eligiendo puntos de vista que cubran la mayor cantidad de pared sin inspeccionar,
se orienta hacia ella, se frena y saca la foto. Cada foto suma a la cobertura (cobertura.py) solo si la profundidad confirma que vio la pared.
Termina cuando ningun punto de vista nuevo suma lo suficiente."""
import time

import numpy as np

import cobertura
import exploracion
import planificador
from mapa_ocupacion import RES, UMBRAL_OCC
from planificador import a_celda, celda_mas_cercana, zona_transitable
from slam_vivo import slam_recortar

PASO_POS = 0.3            # m entre posiciones candidatas
N_YAW = 16                # direcciones candidatas en cada posicion
DESFASE_CAMARA = 0.12     # m: la camara va adelantada respecto del centro del robot
MIN_GANANCIA = 12         # tramos-altura nuevos minimos para que valga la pena ir a un punto de vista
PESO_DIST = 0.5           # tramos-altura que "cuesta" cada metro caminado
MAX_INTENTOS = 2          # un tramo que se intento ver 2 veces sin lograrlo se da por no logrado
ANCHO, ALTO, FOCAL = 320, 240, 120.0   # imagen de profundidad de la camara
T_QUIETO = 1.0            # s orientado y quieto antes de sacar la foto
TOL_YAW = np.radians(5.0)
TOL_POS = 0.10            # m: el punto de vista hay que alcanzarlo con esta precision (la ventana de distancia a la pared es de 30 cm)
VEL_ACERCAR = 0.25        # m/s maximo en la aproximacion final al punto de vista
VEL_MIN = 0.15            # m/s minimo mientras falte mas de TOL_POS * 0.7: por debajo de esto la politica de marcha no se mueve
T_MAX_PUNTO = 60.0        # s: si no llega o no se orienta en este tiempo, abandona ese punto de vista


def inspeccion_crear(cfg=None):
    """Estado de la inspeccion. Los parametros del yaml (todos opcionales) deciden cuan a fondo se inspecciona:
      inspeccion_cobertura_objetivo: % de pared inspeccionada al que se da por terminado (sin esto: hasta que ya no queda nada que sume)
      inspeccion_ganancia_min:       tramos-altura nuevos minimos para que valga la pena ir a un punto de vista
      inspeccion_dist_max:           m, distancia maxima de la camara a la pared para dar un tramo por inspeccionado (mas distancia = menos detalle)
      inspeccion_tol_pos:            m, precision con la que hay que llegar al punto de vista
      inspeccion_t_quieto:           s quieto y orientado antes de sacar la foto"""
    global MIN_GANANCIA, TOL_POS, T_QUIETO
    cfg = cfg or {}
    MIN_GANANCIA = cfg.get("inspeccion_ganancia_min", MIN_GANANCIA)
    TOL_POS = cfg.get("inspeccion_tol_pos", TOL_POS)
    T_QUIETO = cfg.get("inspeccion_t_quieto", T_QUIETO)
    cobertura.DIST_MAX = cfg.get("inspeccion_dist_max", cobertura.DIST_MAX)
    return {"vp": None, "n": 0, "intentos": {}, "terminada": False, "listo": False, "t_ok": None,
            "objetivo": cfg.get("inspeccion_cobertura_objetivo"), "t0": None, "t_ult": 0.0}


def pendientes(cob, ins):
    """Tramos-altura sin ver: (puntos (n,3), normales (n,2), indice de parche (n), claves (n), bandas (n), centros de parche (p,2))."""
    nb, zb = cob["nb"], (np.arange(cob["nb"]) + 0.5) * cob["alto"] / cob["nb"]
    pts, nor, pid, cl, ba, cen = [], [], [], [], [], []
    for k, est in cob["estado"].items():
        libres = [b for b in np.nonzero(est == 0)[0] if ins["intentos"].get((k, int(b)), 0) < MAX_INTENTOS]
        if not libres:
            continue
        c, n = cob["geom"][k]
        for b in libres:
            pts.append([c[0], c[1], zb[b]])
            nor.append(n)
            pid.append(len(cen))
            cl.append(k)
            ba.append(int(b))
        cen.append(c)
    if not pts:
        return None
    return np.array(pts), np.array(nor), np.array(pid), cl, ba, np.array(cen)


def visibilidad(L, lo, pos, cen, muestras=14):
    """(m, p) True si desde la posicion i se ve el parche j sin una pared de por medio (segun el mapa)."""
    ocup = L > UMBRAL_OCC
    d = cen[None, :, :] - pos[:, None, :]
    largo = np.maximum(np.linalg.norm(d, axis=2), 1e-6)
    hasta = np.clip(1.0 - 0.3 / largo, 0.0, 1.0)              # no cuenta el ultimo tramo: es la pared misma
    t = np.linspace(0.02, 1.0, muestras)[None, None, :] * hasta[:, :, None]
    p = pos[:, None, None, :] + t[..., None] * d[:, :, None, :]
    c = np.floor((p - lo) / RES).astype(int)
    dentro = (c[..., 0] >= 0) & (c[..., 0] < L.shape[1]) & (c[..., 1] >= 0) & (c[..., 1] < L.shape[0])
    cx, cy = np.clip(c[..., 0], 0, L.shape[1] - 1), np.clip(c[..., 1], 0, L.shape[0] - 1)
    tapado = (ocup[cy, cx] & dentro).any(axis=2)
    return ~tapado


def elegir_punto_de_vista(L, lo, cob, ins, pos, cam_z):
    """El mejor punto de vista (x, y, yaw, ganancia, esperados) o None."""
    pen = pendientes(cob, ins)
    if pen is None:
        return None
    P, N, pid, cl, ba, cen = pen
    trans, _ = zona_transitable(L)
    ci = a_celda(pos, lo)
    ci2 = ci if trans[ci[1], ci[0]] else celda_mas_cercana(trans, ci)
    if ci2 is None:
        return None
    costo = exploracion.costos_desde(trans, ci2)
    paso = max(1, int(PASO_POS / RES))
    mask = np.zeros(trans.shape, dtype=bool)
    mask[::paso, ::paso] = True
    fs, cs = np.nonzero(mask & trans & np.isfinite(costo))
    if len(fs) == 0:
        return None
    cand = lo + (np.stack([cs, fs], axis=1) + 0.5) * RES
    dist_m = costo[fs, cs] * RES
    vis = visibilidad(L, lo, cand, cen)[:, pid]                   # (m, n)
    mejor, mejor_neto = None, -np.inf
    for k in range(N_YAW):
        yaw = 2 * np.pi * k / N_YAW
        c, s = np.cos(yaw), np.sin(yaw)
        cx, cy = cand[:, 0] + DESFASE_CAMARA * c, cand[:, 1] + DESFASE_CAMARA * s
        rx, ry, rz = P[None, :, 0] - cx[:, None], P[None, :, 1] - cy[:, None], P[None, :, 2] - cam_z
        fwd = c * rx + s * ry
        lat = -s * rx + c * ry
        fw = np.where(fwd > 1e-6, fwd, 1.0)
        u, v = ANCHO / 2 - FOCAL * lat / fw, ALTO / 2 - FOCAL * rz / fw
        dist3 = np.maximum(np.sqrt(rx ** 2 + ry ** 2 + rz ** 2), 1e-9)
        cos_inc = -(rx * N[None, :, 0] + ry * N[None, :, 1]) / dist3
        ok = (vis & (fwd > 1e-6) & (fwd >= cobertura.DIST_MIN) & (fwd <= cobertura.DIST_MAX)
              & (u >= cobertura.MARGEN_PX) & (u < ANCHO - cobertura.MARGEN_PX) & (v >= cobertura.MARGEN_PX) & (v < ALTO - cobertura.MARGEN_PX)
              & (cos_inc >= np.cos(np.radians(cobertura.ANGULO_MAX))))
        ganancia = ok.sum(axis=1)
        neto = ganancia - PESO_DIST * dist_m
        i = int(neto.argmax())
        if ganancia[i] >= MIN_GANANCIA and neto[i] > mejor_neto:
            mejor_neto = neto[i]
            mejor = (cand[i].copy(), yaw, int(ganancia[i]), [(cl[j], ba[j]) for j in np.nonzero(ok[i])[0]])
    if mejor is None:
        return None
    return mejor[0], mejor[1], mejor[2], mejor[3]


def inspeccion_decidir(slam, cob, ins, pos, cam_z, t):
    """Se llama cada segundo. Devuelve (waypoints nuevos o None, mensaje). Elige un punto de vista solo si no hay uno en curso."""
    if ins["t0"] is None:
        ins["t0"] = t
    ins["t_ult"] = t
    if ins["vp"] is not None:
        if t - ins["vp"]["t0"] > T_MAX_PUNTO:
            for e in ins["vp"]["esperados"]:
                ins["intentos"][e] = ins["intentos"].get(e, 0) + 1
            ins["vp"], ins["listo"], ins["t_ok"] = None, False, None
            return None, "punto de vista abandonado (no llego u orientarse llevo demasiado)"
        return None, ""
    t0 = time.perf_counter()
    L, lo = slam_recortar(slam, margen=40)
    cobertura.cob_registrar(cob, L, lo)   # las paredes del mapa actual, todavia sin ver
    pct = cobertura.cob_resumen(cob)[0]
    if ins["objetivo"] is not None and pct >= ins["objetivo"]:
        ins["terminada"] = True
        return None, f"cobertura objetivo alcanzada ({pct:.0f}% de {ins['objetivo']:.0f}%): barrido terminado"
    r = elegir_punto_de_vista(L, lo, cob, ins, pos, cam_z)
    if r is None:
        ins["terminada"] = True
        return None, "no queda pared cubrible con un punto de vista nuevo: inspeccion terminada"
    xy, yaw, ganancia, esperados = r
    wp, _, _ = planificador.planificar(L, lo, pos, xy)
    if wp is None:
        for e in esperados:
            ins["intentos"][e] = ins["intentos"].get(e, 0) + 1
        return None, f"sin ruta hasta ({xy[0]:.1f}, {xy[1]:.1f}): se descarta"
    if len(wp) == 0:
        wp = np.array([xy])
    ins["vp"] = {"xy": xy, "yaw": yaw, "esperados": esperados, "t0": t}
    ins["listo"], ins["t_ok"] = False, None
    return wp, (f"punto {ins['n'] + 1}: ir a ({xy[0]:.1f}, {xy[1]:.1f}) mirando a {np.degrees(yaw):.0f} grados, "
                f"suma ~{ganancia} tramos ({1000 * (time.perf_counter() - t0):.0f} ms)")


def inspeccion_error_yaw(ins, yaw):
    """Error de rumbo (rad) hacia el punto de vista en curso."""
    return float(np.arctan2(np.sin(ins["vp"]["yaw"] - yaw), np.cos(ins["vp"]["yaw"] - yaw)))


def inspeccion_acercar(ins, pos, yaw, kp):
    """Velocidad (vx, vy) en el marco del robot para llegar al punto exacto; el control de waypoints se conforma con 40 cm."""
    d = ins["vp"]["xy"] - np.asarray(pos)
    vx, vy = np.cos(yaw) * d[0] + np.sin(yaw) * d[1], -np.sin(yaw) * d[0] + np.cos(yaw) * d[1]
    v = np.array([kp * vx, kp * vy])
    n = np.linalg.norm(v)
    if np.linalg.norm(d) < TOL_POS * 0.7:
        return np.zeros(2)
    if n < VEL_MIN:
        return v / max(n, 1e-9) * VEL_MIN
    return v if n <= VEL_ACERCAR else v * VEL_ACERCAR / n


def inspeccion_orientar(ins, pos, yaw, t):
    """Llamar en cada paso mientras el robot esta en el punto de vista: devuelve el error de rumbo y marca 'listo' cuando lleva T_QUIETO bien ubicado y orientado."""
    err = inspeccion_error_yaw(ins, yaw)
    if abs(err) < TOL_YAW and np.linalg.norm(ins["vp"]["xy"] - np.asarray(pos)) < TOL_POS:
        if ins["t_ok"] is None:
            ins["t_ok"] = t
        elif t - ins["t_ok"] >= T_QUIETO:
            ins["listo"] = True
    else:
        ins["t_ok"] = None
    return err


def inspeccion_foto_hecha(ins, cob):
    """Despues de sacar la foto y sumarla a la cobertura: anota que se intento y libera el punto de vista."""
    for e in ins["vp"]["esperados"]:
        if cob["estado"][e[0]][e[1]] == 0:
            ins["intentos"][e] = ins["intentos"].get(e, 0) + 1
    ins["n"] += 1
    ins["vp"], ins["listo"], ins["t_ok"] = None, False, None


def inspeccion_texto(ins, cob):
    """Resumen: fotos sacadas y tramos que se intentaron ver sin lograrlo."""
    fallidos = sum(1 for (k, b), n in ins["intentos"].items() if n >= MAX_INTENTOS and k in cob["estado"] and cob["estado"][k][b] == 0)
    dur = f" en {ins['t_ult'] - ins['t0']:.0f} s" if ins["t0"] is not None else ""
    return f"{ins['n']} fotos de inspeccion{dur}; {fallidos} tramos-altura intentados sin lograr verlos"
