"""Danos: detector SIMULADO con la misma entrada y salida que el del back (una foto -> "hay dano" si/no, con una probabilidad), ubicacion de cada
hallazgo sobre la pared y agrupado de los que son el mismo dano. El modelo del back no marca DONDE esta el dano dentro de la foto: aca eso se resuelve
dividiendo la foto en una grilla (columnas x filas) y preguntando por cada recorte, y cada recorte positivo se lleva a la pared con la imagen de
profundidad y la pose estimada del robot (la del SLAM, con sus errores). El veredicto sale de los danos reales declarados en el yaml (con la pose real
de la camara, que el robot no conoce) y NO usa el modelo del back."""
import math
import os

import cv2
import numpy as np

from mapa_ocupacion import RES, UMBRAL_OCC
from slam_vivo import slam_recortar

CARAS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}   # hacia donde mira la cara de la pared (hacia donde esta el robot)
MUESTRAS = 9            # puntos por lado con los que se revisa si un dano se ve (9 x 9 sobre su rectangulo)
TOL_PROF = 0.35         # m: la profundidad medida tiene que coincidir con la esperada para dar por visto un punto (si no, hay algo adelante)
ANGULO_MAX = 75.0       # grados entre la normal de la pared y la direccion hacia la camara: de mas costado no se ve el dano
FRAC_MIN = 0.002        # fraccion de la imagen (o del recorte) por debajo de la cual el dano no se ve
PASO_PX = 6             # se mira un pixel de cada 6 de la imagen de profundidad para ubicar la pared
TOL_PLANO = 0.12        # m: un punto esta "en la pared" si queda a menos de esto del plano
MIN_PUNTOS = 100        # puntos minimos para aceptar un plano de pared en la foto
PROF_MAX = 6.0          # m: mas lejos que esto no se usa para ubicar la pared
ALTO_MIN_PLANO = 1.2    # m: un plano que en la foto no se extiende al menos esto en altura es un objeto bajo (mueble, caja, columna baja), no una pared
PAD = 0.06              # m: lo que se agranda cada zona (el tamano de un pixel de profundidad)
HUECO_UNION = 0.05      # m: dos zonas de la misma pared a menos de esto (o que se pisan) son el mismo hallazgo
UTIL = 0.85             # que parte del campo visual se usa para ver un dano entero (el resto es margen)
D_MIN, D_CERCA, D_MAX_FINA = 0.8, 1.0, 2.6   # m: distancia minima a la pared, distancia de la foto "de cerca", maxima a la que todavia sirve la foto


def danos_crear(cfg):
    """Estado del detector simulado, o None si el yaml no lo pide ('detector_simulado: true')."""
    if not cfg.get("detector_simulado"):
        return None
    reales = []
    for i, d in enumerate(cfg.get("danos") or []):
        cara = str(d["cara"]).strip()
        if cara not in CARAS:
            raise SystemExit(f"[danos] danos[{i}]: la cara '{cara}' no vale; tiene que ser '+x', '-x', '+y' o '-y' (hacia donde mira la cara de la pared)")
        n = np.array(CARAS[cara])
        reales.append({"i": i, "tipo": str(d.get("tipo", "dano")), "c": np.array([float(d["x"]), float(d["y"]), float(d["z"])]),
                       "n": n, "t": np.array([-n[1], n[0]]), "w": float(d["ancho"]), "h": float(d["alto"])})
    grilla = cfg.get("detector_grilla") or [3, 2]
    return {"reales": reales, "p_max": float(cfg.get("detector_prob_max", 0.9)), "area_ref": float(cfg.get("detector_area_ref", 0.04)),
            "falsos": float(cfg.get("detector_falsos", 0.02)), "grilla": (int(grilla[0]), int(grilla[1])),
            "rng": np.random.default_rng(int(cfg.get("detector_semilla", 0))), "alto": float(cfg.get("alto_pared", 2.8)),
            "fotos": [], "dets": [], "hallazgos": []}


def _fraccion_visible(d, cam_pos, R_cam, depth, f, ventana):
    """Que fraccion de la ventana (u0, v0, u1, v1 en pixeles) ocupa el dano d, contando solo lo que se ve de frente y sin nada adelante.
    cam_pos y R_cam: pose REAL de la camara en el mundo (camara MuJoCo: x der, y arriba, mira por -z)."""
    alto_img, ancho_img = depth.shape
    u0, v0, u1, v1 = ventana
    a = np.linspace(-0.5, 0.5, MUESTRAS)
    s, z = np.meshgrid(a * d["w"], a * d["h"])
    P = np.stack([d["c"][0] + s * d["t"][0], d["c"][1] + s * d["t"][1], d["c"][2] + z], axis=-1).reshape(-1, 3)
    dA = d["w"] * d["h"] / MUESTRAS ** 2
    rel = P - cam_pos
    pc = rel @ R_cam
    fwd = -pc[:, 2]
    adelante = fwd > 0.2
    fw = np.where(adelante, fwd, 1.0)
    u = ancho_img / 2 + f * pc[:, 0] / fw
    v = alto_img / 2 - f * pc[:, 1] / fw
    dentro = adelante & (u >= u0 + 1) & (u < u1 - 1) & (v >= v0 + 1) & (v < v1 - 1)
    dist3 = np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
    cos_inc = -(rel[:, 0] * d["n"][0] + rel[:, 1] * d["n"][1]) / dist3
    cand = dentro & (cos_inc >= math.cos(math.radians(ANGULO_MAX)))
    iu = np.clip(np.round(u).astype(int), 1, ancho_img - 2)
    iv = np.clip(np.round(v).astype(int), 1, alto_img - 2)
    coincide = np.zeros(len(P), dtype=bool)
    for dv in (-1, 0, 1):
        for du in (-1, 0, 1):
            coincide |= np.abs(depth[iv + dv, iu + du] - fwd) <= TOL_PROF
    vis = cand & coincide
    eje = -R_cam[:, 2]
    cos_eje = abs(eje[0] * d["n"][0] + eje[1] * d["n"][1])
    px = f * f * dA * cos_eje / fw ** 2          # pixeles que ocupa cada muestra en la imagen
    return float(px[vis].sum() / ((u1 - u0) * (v1 - v0)))


def _veredicto(dn, cam_pos, R_cam, depth, f, ventana, falsos=None):
    """Lo que diria el detector del back para esa ventana de la imagen: {'hay': bool, 'prob': 0 a 1, 'vistos': [(indice del dano real, fraccion)], 'falso': bool}.
    Mas grande se ve el dano en la ventana, mas probable es detectarlo; con ninguno a la vista hay una probabilidad chica de falso positivo."""
    ps, vistos = [], []
    for d in dn["reales"]:
        fr = _fraccion_visible(d, cam_pos, R_cam, depth, f, ventana)
        if fr >= FRAC_MIN:
            ps.append(dn["p_max"] * min(1.0, math.sqrt(fr / dn["area_ref"])))
            vistos.append((d["i"], fr))
    rng = dn["rng"]
    pr = 1.0 - float(np.prod([1.0 - p for p in ps])) if ps else 0.0
    hay, falso = bool(rng.random() < pr), False
    if not hay and not ps and rng.random() < (dn["falsos"] if falsos is None else falsos):
        hay, falso = True, True
    prob = float(np.clip(0.5 + 0.5 * pr + rng.normal(0, 0.05), 0.5, 0.99)) if hay else float(rng.uniform(0.02, 0.35))
    return {"hay": hay, "prob": prob, "vistos": vistos, "falso": falso}


def _planos(cam_pos, R_cam, depth, f, alto, rng):
    """Paredes que mira la foto, con la pose ESTIMADA: hasta 2 planos verticales (RANSAC sobre los puntos 3D de la profundidad, sin piso ni techo).
    Cada uno: normal hacia el robot, un punto, tangente y los puntos con su pixel (u, v)."""
    alto_img, ancho_img = depth.shape
    v, u = np.mgrid[PASO_PX // 2:alto_img:PASO_PX, PASO_PX // 2:ancho_img:PASO_PX]
    z_prof = depth[v, u]
    ok = np.isfinite(z_prof) & (z_prof > 0.3) & (z_prof < PROF_MAX)
    u, v, z_prof = u[ok].astype(float), v[ok].astype(float), z_prof[ok]
    pc = np.stack([(u - ancho_img / 2) / f * z_prof, -(v - alto_img / 2) / f * z_prof, -z_prof], axis=1)
    pw = np.asarray(cam_pos) + pc @ np.asarray(R_cam).T
    util = (pw[:, 2] > 0.15) & (pw[:, 2] < alto + 0.1)
    pw, u, v = pw[util], u[util], v[util]
    planos = []
    for _ in range(2):
        if len(pw) < MIN_PUNTOS:
            break
        xy = pw[:, :2]
        mejor = None
        for _ in range(60):
            i, j = rng.integers(0, len(xy), 2)
            e = xy[j] - xy[i]
            largo = float(np.hypot(e[0], e[1]))
            if largo < 0.3:
                continue
            n = np.array([-e[1], e[0]]) / largo
            cuenta = int((np.abs((xy - xy[i]) @ n) < TOL_PLANO).sum())
            if mejor is None or cuenta > mejor[0]:
                mejor = (cuenta, n, xy[i].copy())
        if mejor is None or mejor[0] < MIN_PUNTOS:
            break
        _, n, p = mejor
        inl = np.abs((xy - p) @ n) < TOL_PLANO
        c = xy[inl].mean(axis=0)
        t = np.linalg.svd(xy[inl] - c, full_matrices=False)[2][0]    # ajuste fino con los puntos de la pared
        n = np.array([-t[1], t[0]])
        inl = np.abs((xy - c) @ n) < TOL_PLANO
        if inl.sum() < MIN_PUNTOS:
            break
        if n @ (np.asarray(cam_pos)[:2] - c) < 0:
            n = -n
        if float(np.ptp(pw[inl][:, 2])) >= ALTO_MIN_PLANO:
            planos.append({"n": n, "t": np.array([-n[1], n[0]]), "p": c, "pts": pw[inl], "u": u[inl], "v": v[inl]})
        pw, u, v = pw[~inl], u[~inl], v[~inl]
    return planos


def _zona(plano, ventana, alto):
    """Parte de la pared (rango a lo largo s y altura z) que cae dentro de la ventana de la imagen; None si casi no hay pared ahi."""
    u0, v0, u1, v1 = ventana
    m = (plano["u"] >= u0) & (plano["u"] < u1) & (plano["v"] >= v0) & (plano["v"] < v1)
    if m.sum() < 6:
        return None
    pts = plano["pts"][m]
    s = (pts[:, :2] - plano["p"]) @ plano["t"]
    return {"n": plano["n"], "t": plano["t"], "p": plano["p"], "s": (float(s.min()) - PAD, float(s.max()) + PAD),
            "z": (max(0.0, float(pts[:, 2].min()) - PAD), min(alto, float(pts[:, 2].max()) + PAD)), "k": int(m.sum())}


def _centro(z):
    """x, y en el marco del mapa del centro de la zona."""
    return z["p"] + z["t"] * 0.5 * (z["s"][0] + z["s"][1])


def _cara_texto(n):
    return ("+x" if n[0] > 0.7 else "-x") if abs(n[0]) > abs(n[1]) else ("+y" if n[1] > 0 else "-y")


def dn_foto(dn, slam, data, cam_id, torso_id, depth, f, n_foto, t_sim):
    """Una foto de inspeccion: veredicto del detector simulado; si da positivo, se divide en la grilla, cada recorte positivo se ubica en la pared
    y todo queda como detecciones para agrupar al final. Devuelve la linea para la consola."""
    alto_img, ancho_img = depth.shape
    cam_pos, R_cam = data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy()
    v = _veredicto(dn, cam_pos, R_cam, depth, f, (0, 0, ancho_img, alto_img))
    dn["fotos"].append({"n": n_foto, "t": t_sim, "hay": v["hay"], "prob": v["prob"], "falso": v["falso"], "vistos": [i for i, _ in v["vistos"]]})
    a_la_vista = ("; reales a la vista: " + ", ".join(f"#{i} ({100 * fr:.1f}% de la imagen)" for i, fr in v["vistos"])) if v["vistos"] else ""
    if not v["hay"]:
        return f"[g1_nav] detector (simulado): foto {n_foto} sin dano ({100 * v['prob']:.0f}% de dano){a_la_vista}"
    R_est = slam["R"] @ (data.xmat[torso_id].reshape(3, 3).T @ R_cam)          # la misma pose estimada que usa la cobertura
    pos_est = np.array([slam["tr"][0], slam["tr"][1], cam_pos[2]])
    planos = _planos(pos_est, R_est, depth, f, dn["alto"], dn["rng"])
    cols, filas = dn["grilla"]
    zonas, positivos = [], 0
    for r in range(filas):
        for c in range(cols):
            vent = (c * ancho_img / cols, r * alto_img / filas, (c + 1) * ancho_img / cols, (r + 1) * alto_img / filas)
            if _veredicto(dn, cam_pos, R_cam, depth, f, vent, falsos=dn["falsos"] / (cols * filas))["hay"]:   # entre todos los recortes, la misma probabilidad de falso por foto
                positivos += 1
                zt = [z for z in (_zona(pl, vent, dn["alto"]) for pl in planos) if z is not None]
                if os.environ.get("DANOS_DEBUG"):
                    print(f"   [dbg] foto {n_foto} recorte {r},{c} planos={[(round(float(q['n'][0]),2), round(float(q['n'][1]),2), len(q['pts'])) for q in planos]} zonas={[(q['k'], tuple(round(x,2) for x in q['z']), tuple(round(x,2) for x in q['s'])) for q in zt]}")
                if zt:
                    zonas.append(max(zt, key=lambda q: q["k"]))                   # el recorte se atribuye a la pared que mas ocupa en ese recorte
    if not zonas:                                                              # positiva la foto entera pero ningun recorte: se queda con toda la pared que se ve
        zonas = [z for z in (_zona(pl, (0, 0, ancho_img, alto_img), dn["alto"]) for pl in planos) if z is not None]
    for z in zonas:
        dn["dets"].append({"foto": n_foto, "prob": v["prob"], "falso": v["falso"], "zona": z})
    if not zonas:
        return f"[g1_nav] detector (simulado): foto {n_foto} DANO ({100 * v['prob']:.0f}%), sin pared plana para ubicarlo{a_la_vista}"
    z0, z1 = min(z["z"][0] for z in zonas), max(z["z"][1] for z in zonas)
    c = _centro(zonas[0])
    return (f"[g1_nav] detector (simulado): foto {n_foto} DANO ({100 * v['prob']:.0f}%), {positivos} de {cols * filas} recortes; "
            f"pared {_cara_texto(zonas[0]['n'])} cerca de ({c[0]:.1f}, {c[1]:.1f}), altura {z0:.1f}-{z1:.1f} m{a_la_vista}")


def _en_marco(a, b):
    """Rangos (s0, s1, z0, z1) de la zona b expresados en el marco de la zona a, o None si no son de la misma pared."""
    if a["n"] @ b["n"] < math.cos(math.radians(20)) or abs((b["p"] - a["p"]) @ a["n"]) > 0.25:
        return None
    off, k = (b["p"] - a["p"]) @ a["t"], b["t"] @ a["t"]
    s = sorted((off + k * b["s"][0], off + k * b["s"][1]))
    return s[0], s[1], b["z"][0], b["z"][1]


def _se_juntan(a, b):
    r = _en_marco(a, b)
    if r is None:
        return False
    return min(a["s"][1], r[1]) - max(a["s"][0], r[0]) > -HUECO_UNION and min(a["z"][1], r[3]) - max(a["z"][0], r[2]) > -HUECO_UNION


def dn_agrupar(dn):
    """Junta las zonas positivas que son el mismo dano (misma pared y se tocan o se pisan). Cada hallazgo: la zona comun a todas las fotos que lo vieron
    (si no hay una comun, la que las abarca a todas), con cuantas fotos lo vieron y la mayor probabilidad. Queda en dn['hallazgos'], de mas a menos respaldado."""
    zs = [d["zona"] for d in dn["dets"]]
    padre = list(range(len(zs)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            if _se_juntan(zs[i], zs[j]):
                padre[raiz(j)] = raiz(i)
    grupos = {}
    for i in range(len(zs)):
        grupos.setdefault(raiz(i), []).append(i)
    res = []
    for idx in grupos.values():
        a = zs[idx[0]]
        rangos = [(a["s"][0], a["s"][1], a["z"][0], a["z"][1])] + [_en_marco(a, zs[j]) for j in idx[1:]]
        rangos = [r for r in rangos if r is not None]                           # (una zona unida por una cadena puede no ser comparable con la primera)
        s0, s1 = max(r[0] for r in rangos), min(r[1] for r in rangos)
        z0, z1 = max(r[2] for r in rangos), min(r[3] for r in rangos)
        if s1 - s0 < 0.2 or z1 - z0 < 0.2:                                      # no hay zona comun: la que abarca a todas
            s0, s1 = min(r[0] for r in rangos), max(r[1] for r in rangos)
            z0, z1 = min(r[2] for r in rangos), max(r[3] for r in rangos)
        fotos = sorted({dn["dets"][j]["foto"] for j in idx})
        res.append({"n": a["n"], "t": a["t"], "p": a["p"], "s": (s0, s1), "z": (z0, z1), "fotos": fotos, "apoyo": len(fotos),
                    "prob": max(dn["dets"][j]["prob"] for j in idx)})
    res.sort(key=lambda h: (-h["apoyo"], -h["prob"]))
    for k, h in enumerate(res, 1):
        h["id"] = k
    dn["hallazgos"] = res
    return res


def dn_distancia_para_ver(z0, z1, ancho, cam_z, f, forma=(240, 320)):
    """A que distancia de la pared tiene que pararse el robot para ver ENTERO un dano que va de la altura z0 a z1 y mide 'ancho' a lo largo.
    La camara va fija a cam_z de altura y no se inclina: a distancia d solo ve de cam_z - tv*d a cam_z + tv*d (tv: tangente util del campo visual vertical).
    Devuelve {'d': m, 'altura': 'bajo'|'medio'|'alto', 'entra': si d <= D_MAX_FINA, 'px_por_m': detalle a esa distancia}."""
    tv, th = UTIL * (forma[0] / 2) / f, UTIL * (forma[1] / 2) / f
    d = max(D_MIN, (z1 - cam_z) / tv, (cam_z - z0) / tv, (ancho / 2) / th)
    zc = 0.5 * (z0 + z1)
    return {"d": d, "altura": "bajo" if zc < 0.8 else ("medio" if zc < 1.5 else "alto"), "entra": d <= D_MAX_FINA, "px_por_m": f / d}


def _a_marco_mapa(slam, p_mundo, n_mundo):
    """Punto y normal del marco del mundo al del mapa del SLAM (origen en el torso al arrancar, con el rumbo con el que arranco)."""
    xy0, yaw0 = slam["real0"]
    c, s = math.cos(yaw0), math.sin(yaw0)
    R = np.array([[c, s], [-s, c]])
    return R @ (np.asarray(p_mundo, dtype=float) - xy0), R @ np.asarray(n_mundo, dtype=float)


def _real_en_mapa(dn, slam):
    """Los danos reales con centro, normal y tangente en el marco del mapa (para comparar con lo que se encontro)."""
    out = []
    for d in dn["reales"]:
        c, n = _a_marco_mapa(slam, d["c"][:2], d["n"])
        out.append({"i": d["i"], "tipo": d["tipo"], "c": c, "z": float(d["c"][2]), "n": n, "t": np.array([-n[1], n[0]]), "w": d["w"], "h": d["h"]})
    return out


def _coincide(h, r, margen=0.3):
    """El hallazgo h (zona en la pared) contiene el centro del dano real r."""
    if h["n"] @ r["n"] < math.cos(math.radians(25)) or abs((r["c"] - h["p"]) @ h["n"]) > 0.5:
        return False
    s = (r["c"] - h["p"]) @ h["t"]
    return h["s"][0] - margen <= s <= h["s"][1] + margen and h["z"][0] - margen <= r["z"] <= h["z"][1] + margen


def dn_evaluar(dn, slam):
    """Compara hallazgos con danos reales: (reales encontrados, reales, hallazgos sin dano real). Solo para validar: el robot no conoce los reales."""
    reales = _real_en_mapa(dn, slam)
    enc = [r for r in reales if any(_coincide(h, r) for h in dn["hallazgos"])]
    falsos = [h for h in dn["hallazgos"] if not any(_coincide(h, r) for r in reales)]
    return enc, reales, falsos


def dn_texto_hallazgo(h, cam_z, f):
    c = h["p"] + h["t"] * 0.5 * (h["s"][0] + h["s"][1])
    ver = dn_distancia_para_ver(h["z"][0], h["z"][1], h["s"][1] - h["s"][0], cam_z, f)
    entra = "se ve entero de una foto" if ver["entra"] else "no entra en una sola foto: hay que sacar varias"
    return (f"hallazgo {h['id']}: pared {_cara_texto(h['n'])} en ({c[0]:.1f}, {c[1]:.1f}), ancho {h['s'][1] - h['s'][0]:.1f} m, altura {h['z'][0]:.1f}-{h['z'][1]:.1f} m "
            f"({ver['altura']}); visto en {h['apoyo']} foto(s), prob {100 * h['prob']:.0f}%; parado a {ver['d']:.1f} m de la pared, {entra}")


def dn_dibujar(dn, slam, L, lo, ruta_png, px_por_m=40):
    """Mapa con los danos reales (rojo), cada zona positiva de cada foto (azul fino) y los hallazgos agrupados (verde si coincide con un dano real, naranja si no)."""
    fpx = px_por_m * RES
    img = np.full(L.shape + (3,), 160, np.uint8)
    img[L < -0.2] = (255, 255, 255)
    img[L > UMBRAL_OCC] = (0, 0, 0)
    img = cv2.resize(img, None, fx=fpx, fy=fpx, interpolation=cv2.INTER_NEAREST)[::-1].copy()
    alto_img = img.shape[0]

    def px(a):
        q = np.round((np.asarray(a, dtype=float) - lo) / RES * fpx).astype(int)
        return int(q[0]), int(alto_img - 1 - q[1])

    def seg(p, n, t, s0, s1, color, grosor, desvio):
        cv2.line(img, px(p + desvio * n + s0 * t), px(p + desvio * n + s1 * t), color, grosor)

    reales = _real_en_mapa(dn, slam)
    for r in reales:
        seg(r["c"], r["n"], r["t"], -r["w"] / 2, r["w"] / 2, (0, 0, 220), 6, 0.06)
    for d in dn["dets"]:
        z = d["zona"]
        seg(z["p"], z["n"], z["t"], z["s"][0], z["s"][1], (230, 120, 0), 1, 0.22)
    for h in dn["hallazgos"]:
        ok = any(_coincide(h, r) for r in reales)
        seg(h["p"], h["n"], h["t"], h["s"][0], h["s"][1], (0, 170, 0) if ok else (0, 140, 255), 4, 0.14)
        cv2.putText(img, str(h["id"]), px(h["p"] + 0.35 * h["n"] + 0.5 * (h["s"][0] + h["s"][1]) * h["t"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 90, 0) if ok else (0, 90, 200), 1, cv2.LINE_AA)
    cv2.imwrite(ruta_png, img)


def dn_cierre(dn, slam, cam_z, f, ruta_base):
    """Al terminar el barrido: agrupa, compara con los danos reales, guarda el mapa y devuelve el texto para la consola."""
    hall = dn_agrupar(dn)
    enc, reales, falsos = dn_evaluar(dn, slam)
    pos = sum(1 for x in dn["fotos"] if x["hay"])
    lineas = [f"[g1_nav] detector (simulado): {len(dn['fotos'])} fotos, {pos} con dano; {len(hall)} hallazgos agrupados; "
              f"danos reales del yaml encontrados: {len(enc)} de {len(reales)}; hallazgos sin dano real: {len(falsos)}"]
    lineas += ["[g1_nav] " + dn_texto_hallazgo(h, cam_z, f) for h in hall]
    perdidos = [r for r in reales if r not in enc]
    if perdidos:
        lineas.append("[g1_nav] sin encontrar: " + "; ".join(f"{r['tipo']} #{r['i']} en ({r['c'][0]:.1f}, {r['c'][1]:.1f}), z {r['z']:.1f}" for r in perdidos) + " (coordenadas del mapa del robot)")
    L, lo = slam_recortar(slam)
    dn_dibujar(dn, slam, L, lo, ruta_base + "_danos.png")
    lineas.append(f"[g1_nav] mapa de danos: {ruta_base}_danos.png (rojo = dano real del yaml, azul = zona de cada foto positiva, verde = hallazgo con dano real, naranja = sin dano real)")
    return "\n".join(lineas)
