"""Plano y recorrido para el front. El front dibuja el mapa SOLO con 'paredes' (celdas ocupadas) del POST /room y el camino con POST /pose.
El mapa que se manda es el que arma el SLAM del robot, en el mismo marco que los hallazgos y la posicion (origen donde arranca el robot).
Solo funciones: el estado es un dict que crea plano_crear."""
import threading
import time

import numpy as np
import requests

import slam_vivo
from mapa_ocupacion import RES, UMBRAL_OCC

POSE_CADA_S = 0.5      # segundos de simulacion entre posiciones enviadas (el back espera una cada medio segundo)
MIN_BARRIDOS = 10      # no se manda un mapa hasta tener al menos tantos barridos


def plano_crear(cfg, room, url_room):
    """room: lo que devolvio calcular_room_layout (o None si no se pudo calcular); url_room: la URL de POST /room.
    Devuelve None si el yaml tiene 'plano_al_back: false'."""
    if not cfg.get("plano_al_back", True):
        return None
    return {"room": room, "url_room": url_room, "url_pose": url_room.rsplit("/", 1)[0] + "/pose",
            "cada": float(cfg.get("plano_cada_s", 30)), "pose": bool(cfg.get("pose_al_back", True)),
            "t_room": None, "t_pose": None, "n_ult": -1, "completo": False, "hilo_room": None, "hilo_pose": None,
            "avisado": set(), "enviados": 0}


def plano_paredes(sl):
    """Las celdas ocupadas del mapa del SLAM (paredes y objetos, con los obstaculos bajos que vio la camara) como las pide el front:
    {resolucion, celdas: [[i, j], ...]}; la celda (i, j) cubre x de i*resolucion a (i+1)*resolucion, y lo mismo en y. None si todavia no hay nada."""
    if sl is None or not np.any(sl["L"] != 0):
        return None
    L, lo = slam_vivo.slam_recortar(sl, bajos=True)
    fs, cs = np.nonzero(L > UMBRAL_OCC)
    if len(fs) == 0:
        return None
    i0, j0 = np.round(lo / RES).astype(int)    # el origen de la grilla es siempre multiplo de RES
    return {"resolucion": RES, "celdas": np.stack([cs + i0, fs + j0], axis=1).tolist()}


def _room_completo(pl, paredes):
    """El cuerpo del POST /room: el layout del escenario (si se pudo calcular) mas las paredes; si no, los limites salen del propio mapa."""
    if pl["room"] is not None:
        room = dict(pl["room"])
    else:
        c = np.array(paredes["celdas"])
        x0, y0 = c.min(axis=0) * RES
        x1, y1 = (c.max(axis=0) + 1) * RES
        room = {"x_min": float(x0), "x_max": float(x1), "y_min": float(y0), "y_max": float(y1), "width": float(x1 - x0), "depth": float(y1 - y0),
                "floor_polygon": [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]}
    room["paredes"] = paredes
    return room


def _post(pl, que, url, cuerpo, timeout=5):
    """Manda y devuelve (codigo HTTP o None, texto del error). Avisa una sola vez por tipo de problema."""
    try:
        r = requests.post(url, json=cuerpo, timeout=timeout)
        if r.status_code != 200 and que not in pl["avisado"]:
            pl["avisado"].add(que)
            print(f"[g1_nav] plano: el back contesto HTTP {r.status_code} a {url}: {r.text[:150]}")
        return r.status_code, ""
    except requests.exceptions.RequestException as e:
        if que not in pl["avisado"]:
            pl["avisado"].add(que)
            print(f"[g1_nav] plano: no se pudo mandar a {url} ({str(e)[:80]}). No aviso de nuevo.")
        return None, str(e)


def _enviar_room(pl, room, n):
    codigo, _ = _post(pl, "room", pl["url_room"], room)
    if codigo == 200:
        pl["enviados"] += 1
        if pl["enviados"] == 1:
            print(f"[g1_nav] plano: {n} celdas de pared enviadas al back (el front las dibuja; se vuelven a mandar cada {pl['cada']:.0f} s mientras crece el mapa)")


def _ocupado(hilo):
    return hilo is not None and hilo.is_alive()


def plano_tick(pl, sl, t, base_pos, yaw, mapa_completo):
    """Se llama en cada paso de control. Manda la posicion cada POSE_CADA_S y las paredes cada 'plano_cada_s' (y apenas el mapa esta completo).
    t: tiempo de simulacion; mapa_completo: True cuando termino la exploracion. No frena el robot: manda en un hilo."""
    if pl is None:
        return
    if pl["pose"] and (pl["t_pose"] is None or t - pl["t_pose"] >= POSE_CADA_S) and not _ocupado(pl["hilo_pose"]):
        pl["t_pose"] = t
        cuerpo = {"x": float(base_pos[0]), "y": float(base_pos[1]), "yaw": float(yaw), "timestamp": time.time()}
        pl["hilo_pose"] = threading.Thread(target=_post, args=(pl, "pose", pl["url_pose"], cuerpo, 2), daemon=True)
        pl["hilo_pose"].start()
    if sl is None or sl["n"] < MIN_BARRIDOS or _ocupado(pl["hilo_room"]):
        return
    recien_completo = bool(mapa_completo) and not pl["completo"]
    cada = pl["cada"] * (4 if pl["completo"] else 1)    # despues de explorar el mapa solo suma los obstaculos bajos que ve la camara: se manda mas espaciado
    if not (pl["t_room"] is None or t - pl["t_room"] >= cada or recien_completo):
        return
    paredes = plano_paredes(sl)
    if paredes is None:
        return
    pl["t_room"] = t
    pl["completo"] = pl["completo"] or recien_completo
    n = len(paredes["celdas"])
    if n == pl["n_ult"] and not recien_completo:      # el mapa no cambio (despues de explorar queda congelado)
        return
    pl["n_ult"] = n
    pl["hilo_room"] = threading.Thread(target=_enviar_room, args=(pl, _room_completo(pl, paredes), n), daemon=True)
    pl["hilo_room"].start()


def plano_final(pl, sl):
    """Al terminar: espera lo que este en vuelo y manda el mapa final. Devuelve el texto para imprimir."""
    if pl is None:
        return ""
    for h in (pl["hilo_room"], pl["hilo_pose"]):
        if _ocupado(h):
            h.join(5)
    paredes = plano_paredes(sl)
    if paredes is None:
        return "[g1_nav] plano: el SLAM no armo ningun mapa, no hay paredes para mandar al front"
    codigo, err = _post(pl, "room", pl["url_room"], _room_completo(pl, paredes), 15)
    n = len(paredes["celdas"])
    if codigo == 200:
        return f"[g1_nav] plano: mapa final enviado al back: {n} celdas de pared de {RES * 100:.0f} cm (el front lo dibuja con la inspeccion que tenga abierta)"
    return f"[g1_nav] plano: NO se pudo mandar el mapa final al back ({'HTTP ' + str(codigo) if codigo else err[:80]}); el front va a quedar con el ultimo que recibio"
