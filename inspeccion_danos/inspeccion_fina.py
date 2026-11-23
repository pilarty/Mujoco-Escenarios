"""Inspeccion fina: despues del barrido, el robot vuelve a cada hallazgo (danos.dn_agrupar), se para a la distancia calculada para verlo, saca varias
fotos (general, de cerca, desde la izquierda y desde la derecha), afina DONDE esta el dano dentro de la pared (con el mismo detector simulado, ahora con
una grilla mas fina y mas cerca), y manda las fotos al back. Al back le llega la posicion del DANO (x, y sobre la pared, en el marco del mapa del
robot), no la del robot: es el dato que el front usa para senalar el dano. El veredicto de "hay dano" del back se ignora; el de clasificacion
(los 'types' de la respuesta) es el que se usa. Todo son funciones sobre un diccionario de estado."""
import base64
import io
import math
import threading
import time

import numpy as np
import requests

import detector_danos as danos
import exploracion
import inspeccion
import planificador
from mapa_ocupacion import RES
from planificador import a_celda, celda_mas_cercana, zona_transitable
from slam_vivo import slam_recortar

PASOS = ("general", "cerca", "izq", "der")
NOMBRE_PASO = {"general": "general", "cerca": "de cerca", "izq": "desde la izquierda", "der": "desde la derecha"}
PESO_PASO = {"general": 1.0, "cerca": 2.0, "izq": 1.0, "der": 1.0}   # peso de cada foto al decidir la clase del hallazgo (la de cerca vale doble)
EXTRA_MAX = 0.8          # m: si el punto de vista ideal no se puede pisar, se prueba mas lejos de la pared hasta esto
MARGEN_UNION = 0.4       # m: recortes positivos a esta distancia de la zona del hallazgo siguen siendo el mismo dano
MARGEN_RECORTE = 0.12    # fraccion de cada lado que se agrega alrededor del dano al recortar la foto
LADO_MIN_PX = 224        # lado minimo del recorte (el back redimensiona a 224 x 224)
ESPERA_BACK_S = 30.0     # s que se espera, al final, a que el back termine de contestar
DESVIOS_GRADOS = (0, 8, -8, 16, -16, 24, -24)   # si el punto ideal no se puede pisar, se prueba girando alrededor del dano estos grados antes de alejarse
ACERCAR_MIN = 0.3        # m: la foto "de cerca" se saca solo si queda al menos esto mas cerca que la general (en danos altos la camara no puede acercarse)
FOTOS_PARA_DESCARTAR = 2 # fotos (general / cerca / de costado) en las que el detector no ve nada en el hallazgo para darlo por no confirmado


def fina_crear(cfg, dn, url, carpeta):
    """Estado de la inspeccion fina, o None si no esta pedida ('inspeccion_fina: true') o no hay detector simulado. Parametros del yaml (todos opcionales):
      fina_pasos:                 fotos por hallazgo y en que orden (general, cerca, izq, der)
      fina_enviar:                cuales de esas fotos se mandan al back (por defecto todas)
      fina_angulo_grados:         angulo respecto de la perpendicular a la pared para las fotos laterales
      fina_cerca_m:               distancia minima de la foto "de cerca" (puede ser mayor si el dano es alto)
      fina_grilla:                [columnas, filas] con que se afina la ubicacion del dano en la foto general y en la de cerca
      fina_recorte:               true: a las fotos que van al back se las recorta alrededor del dano (el back las achica a 224 x 224)
      fina_clasificar_en_back:    false (por defecto): el back no se toca y se usa lo que conteste. true: ademas se manda el campo 'clasificar_siempre' (solo sirve si el back lo tiene)
      fina_max_s:                 s de simulacion maximos para toda la inspeccion fina"""
    if dn is None or not cfg.get("inspeccion_fina"):
        return None
    pasos = [str(p) for p in (cfg.get("fina_pasos") or PASOS)]
    for p in pasos:
        if p not in PASOS:
            raise SystemExit(f"[fina] fina_pasos: '{p}' no vale; las opciones son {', '.join(PASOS)}")
    enviar = [str(p) for p in (cfg.get("fina_enviar") or pasos)]
    grilla = cfg.get("fina_grilla") or [8, 6]
    fi = {"url": url, "carpeta": carpeta, "pasos": pasos, "enviar": enviar, "ang": float(cfg.get("fina_angulo_grados", 30.0)),
          "d_cerca": float(cfg.get("fina_cerca_m", 1.0)), "grilla": (int(grilla[0]), int(grilla[1])), "recorte": bool(cfg.get("fina_recorte", True)),
          "clasificar": bool(cfg.get("fina_clasificar_en_back", False)), "max_s": float(cfg.get("fina_max_s", 900)),
          "iniciada": False, "terminada": False, "cola": [], "act": None, "total": 0, "hechos": [], "hilos": [], "t0": None, "t_fin": None,
          "n_fotos": 0, "lat": [], "errores": 0, "sin_clase": 0, "listos": 0}
    print(f"[g1_nav] inspeccion fina: fotos por hallazgo {', '.join(pasos)}; se mandan al back {', '.join(enviar)}; grilla de afinado {fi['grilla'][0]}x{fi['grilla'][1]}; "
          f"recorte {'SI' if fi['recorte'] else 'no'}; el back se usa tal cual esta (clasifica cuando su detector lo pide); tiempo maximo {fi['max_s']:.0f} s")
    return fi


def _centro(z):
    return z["p"] + z["t"] * 0.5 * (z["s"][0] + z["s"][1])


def _ordenar(hall, pos):
    """Orden de visita: siempre el hallazgo mas cercano al que se esta."""
    resto, orden, aqui = list(range(len(hall))), [], np.asarray(pos, dtype=float)
    while resto:
        i = min(resto, key=lambda q: float(np.linalg.norm(_centro(hall[q]) - aqui)))
        resto.remove(i)
        orden.append(hall[i])
        aqui = _centro(hall[i])
    return orden


def _abrir(fi, h, k, t):
    zona = {"n": h["n"], "t": h["t"], "p": h["p"], "s": tuple(h["s"]), "z": tuple(h["z"])}
    return {"h": h, "k": k, "zona": zona, "zona0": zona, "pasos": list(fi["pasos"]), "fotos": [], "afinada": False, "saltados": [], "t_ini": t, "hilo": None, "resp": None,
            "n_ch": 0, "conf": 0, "descartado": False, "d_general": None}


def _distancia(fi, act, paso, cam_z, f):
    """Distancia de la camara a la pared para ese paso: la que hace falta para ver entera la zona (general) o, mas cerca si se puede, para el resto."""
    z = act["zona"]
    ver = danos.dn_distancia_para_ver(z["z"][0], z["z"][1], z["s"][1] - z["s"][0], cam_z, f)
    d = ver["d"] if paso == "general" else max(fi["d_cerca"], ver["d"])
    return min(d, danos.D_MAX_FINA)


def _punto_de_vista(fi, act, paso, d, trans, costo, lo, Lv):
    """Donde parar al robot (x, y del robot y rumbo) para sacar esa foto, o None. La camara va a d de la pared mirando al centro del dano; si ahi no se puede
    estar (pared mapeada, obstaculo, sin camino) se prueba cada vez mas lejos."""
    z = act["zona"]
    c, n, t = _centro(z), z["n"], z["t"]
    th0 = {"general": 0.0, "cerca": 0.0, "izq": -math.radians(fi["ang"]), "der": math.radians(fi["ang"])}[paso]   # izq: de frente a la pared es del lado -t
    for extra in np.arange(0.0, EXTRA_MAX + 1e-9, 0.1):
        for dg in DESVIOS_GRADOS:
            th = th0 + math.radians(dg)
            u = math.cos(th) * n + math.sin(th) * t
            robot = c + (d + extra + inspeccion.DESFASE_CAMARA) * u
            ce = a_celda(robot, lo)
            if not (0 <= ce[0] < trans.shape[1] and 0 <= ce[1] < trans.shape[0]):
                continue
            if not trans[ce[1], ce[0]] or not np.isfinite(costo[ce[1], ce[0]]):
                continue
            if not inspeccion.visibilidad(Lv, lo, robot[None, :], c[None, :])[0, 0]:
                continue
            return {"xy": robot, "yaw": math.atan2(-u[1], -u[0]), "d": d + float(extra), "c": c}
    return None


def _recoger(fi):
    """Hallazgos cuyas fotos ya contesto el back: devuelve las lineas para la consola."""
    out = []
    for act in list(fi["hilos"]):
        if act["hilo"].is_alive():
            continue
        fi["hilos"].remove(act)
        out.append(_resumen(fi, act))
    return out


def _clase_final(act):
    """Clase del hallazgo: la de mayor probabilidad ponderada entre las fotos que el back contesto (sin clase, una foto cuenta 0 para todas)."""
    pesos, puntaje, nombres, validas = 0.0, {}, {}, 0
    for fo in act["fotos"]:
        r = fo.get("resp")
        if not r or "types" not in r:
            continue
        validas += 1
        w = PESO_PASO[fo["paso"]]
        pesos += w
        for tp in r["types"]:
            puntaje[tp["type"]] = puntaje.get(tp["type"], 0.0) + w * tp["probability"]
            nombres[tp["type"]] = tp.get("label", tp["type"])
    if not puntaje:
        return None, validas
    k = max(puntaje, key=puntaje.get)
    return {"type": k, "label": nombres[k], "prob": puntaje[k] / pesos, "otras": {nombres[q]: puntaje[q] / pesos for q in puntaje if q != k}}, validas


def _real_del_yaml(dn, slam, zona):
    for r in danos._real_en_mapa(dn, slam):
        if danos._coincide(zona, r):
            return r
    return None


def _resumen(fi, act):
    z = act["zona"]
    c = _centro(z)
    cl, validas = _clase_final(act)
    enviadas = [fo for fo in act["fotos"] if fo.get("resp") is not None]
    if cl is not None:
        texto = f"{cl['label']} {cl['prob']:.0f}%"
        if cl["otras"]:
            texto += " (tambien: " + ", ".join(f"{k} {v:.0f}%" for k, v in sorted(cl["otras"].items(), key=lambda q: -q[1])[:2]) + ")"
    elif validas:
        texto = "el back contesto pero ninguna clase paso su umbral"
        fi["sin_clase"] += 1
    elif enviadas:
        texto = "el back no contesto bien: " + "; ".join(sorted({fo["resp"].get("error", "?") for fo in enviadas}))
    else:
        texto = "sin fotos enviadas"
    act["clase"] = cl
    fi["listos"] += 1
    lat = [fo["resp"]["lat"] for fo in act["fotos"] if fo.get("resp") and "lat" in fo["resp"]]
    ms = f"; el back tardo {sum(lat) / len(lat):.2f} s por foto" if lat else ""
    return (f"fina: hallazgo {act['h']['id']} listo ({fi['listos']} de {fi['total']}): pared en ({c[0]:.2f}, {c[1]:.2f}), ancho {z['s'][1] - z['s'][0]:.1f} m, "
            f"altura {z['z'][0]:.1f}-{z['z'][1]:.1f} m; clasificacion del back: {texto}{ms}")


def _guardar_y_mandar(fi, act, x, y):
    """Hilo: manda al back las fotos del hallazgo (todas con la posicion del dano) y lee de vuelta la clasificacion de cada una."""
    for fo in act["fotos"]:
        if fo["paso"] not in fi["enviar"]:
            continue
        buf = io.BytesIO()
        fo["img"].save(buf, format="PNG")
        payload = {"frame": int(fo["frame"]), "timestamp": float(fo["timestamp"]), "x": float(x), "y": float(y),
                   "imagen_base64": base64.b64encode(buf.getvalue()).decode("utf-8")}
        if fi["clasificar"]:
            payload["clasificar_siempre"] = True
        t0 = time.perf_counter()
        try:
            r = requests.post(fi["url"], json=payload, timeout=30)
            lat = time.perf_counter() - t0
            if r.status_code != 200:
                fo["resp"] = {"error": f"HTTP {r.status_code}", "lat": lat}
                fi["errores"] += 1
                continue
            idr = r.json().get("id")
            reg = next((q for q in requests.get(fi["url"], timeout=30).json() if q.get("id") == idr), None)
            fo["resp"] = {"lat": lat, "types": (reg or {}).get("types", []), "det": (reg or {}).get("detection_probability")}
            fi["lat"].append(lat)
        except (requests.exceptions.RequestException, ValueError) as e:
            fo["resp"] = {"error": "no se pudo conectar con el back (¿esta corriendo uvicorn?)" if isinstance(e, requests.exceptions.ConnectionError) else str(e)[:60]}
            fi["errores"] += 1


def _cerrar(fi, act):
    """Termino el recorrido del hallazgo: manda sus fotos al back con la posicion final del dano (la misma para todas, asi el back las agrupa)."""
    c = _centro(act["zona"])
    act["hilo"] = threading.Thread(target=_guardar_y_mandar, args=(fi, act, c[0], c[1]), daemon=True)
    act["hilo"].start()
    fi["hilos"].append(act)


def fina_decidir(fi, dn, slam, ins, pos, cam_z, f, t):
    """Se llama cada segundo, como inspeccion_decidir. Devuelve (waypoints nuevos o None, lista de mensajes). Elige el proximo punto de vista solo si no hay uno en curso."""
    msgs = _recoger(fi)
    if fi["t0"] is None:
        fi["t0"] = t
    vp = ins["vp"]
    if vp is not None and not vp.get("fino"):
        ins["vp"], ins["listo"], ins["t_ok"] = None, False, None         # un punto del barrido que quedo a medias
        vp = None
    if vp is not None:
        if t - vp["t0"] > inspeccion.T_MAX_PUNTO:
            fi["act"]["saltados"].append(vp["fino"]["paso"])
            msgs.append(f"fina: hallazgo {fi['act']['h']['id']} ({NOMBRE_PASO[vp['fino']['paso']]}): no llego u orientarse llevo demasiado, se salta")
            ins["vp"], ins["listo"], ins["t_ok"] = None, False, None
        else:
            return None, msgs
    if not fi["iniciada"]:
        fi["iniciada"] = True
        hall = danos.dn_agrupar(dn)
        fi["total"] = len(hall)
        fi["cola"] = _ordenar(hall, pos)
        if hall:
            msgs.append(f"fina: {len(hall)} hallazgos para inspeccionar de cerca, en este orden: {', '.join(str(h['id']) for h in fi['cola'])}")
        else:
            msgs.append("fina: el barrido no dejo ningun hallazgo; no hay nada que inspeccionar de cerca")
    vencido = t - fi["t0"] > fi["max_s"]
    if vencido and fi["cola"]:
        msgs.append(f"fina: tiempo maximo ({fi['max_s']:.0f} s) agotado; quedan sin visitar los hallazgos {', '.join(str(h['id']) for h in fi['cola'])}")
        fi["cola"] = []
        if fi["act"] is not None:
            fi["act"]["pasos"] = []
    L = lo = Lv = trans = costo = None
    while True:
        if fi["act"] is None:
            if not fi["cola"]:
                if fi["t_fin"] is None:
                    fi["t_fin"] = t
                if not fi["hilos"] or t - fi["t_fin"] >= ESPERA_BACK_S:   # (espera a que el back termine de contestar las ultimas fotos)
                    fi["terminada"] = True
                return None, msgs
            h = fi["cola"].pop(0)
            fi["act"] = _abrir(fi, h, fi["total"] - len(fi["cola"]), t)
        act = fi["act"]
        if not act["pasos"]:
            fi["hechos"].append(act)
            if act["descartado"]:
                fi["listos"] += 1
                msgs.append(f"fina: hallazgo {act['h']['id']} descartado ({fi['listos']} de {fi['total']}): el detector no vio dano de cerca; no se manda nada al back")
            elif act["fotos"]:
                _cerrar(fi, act)
            else:
                fi["listos"] += 1
                msgs.append(f"fina: hallazgo {act['h']['id']} ({fi['listos']} de {fi['total']}): no se pudo sacar ninguna foto")
            fi["act"] = None
            continue
        paso = act["pasos"].pop(0)
        if L is None:
            L, lo = slam_recortar(slam, margen=40, bajos=True)
            Lv, _ = slam_recortar(slam, margen=40)
            trans, _ = zona_transitable(L, casco=True)
            ci = a_celda(pos, lo)
            ci2 = ci if trans[ci[1], ci[0]] else celda_mas_cercana(trans, ci)
            if ci2 is None:
                msgs.append("fina: el robot esta en una zona sin espacio libre conocido; se corta la inspeccion fina")
                fi["cola"], act["pasos"] = [], []
                continue
            costo = exploracion.costos_desde(trans, ci2)
        d = _distancia(fi, act, paso, cam_z, f)
        if paso == "cerca" and act["d_general"] is not None and d > act["d_general"] - ACERCAR_MIN:
            act["saltados"].append(paso)
            msgs.append(f"fina: hallazgo {act['h']['id']} (de cerca): no se puede acercar mas que {act['d_general']:.1f} m (el dano es alto y la camara no se inclina), se salta")
            continue
        vpn = _punto_de_vista(fi, act, paso, d, trans, costo, lo, Lv)
        if vpn is None:
            act["saltados"].append(paso)
            msgs.append(f"fina: hallazgo {act['h']['id']} ({NOMBRE_PASO[paso]}): no hay donde pararse a {d:.1f} m de la pared, se salta")
            continue
        wp, _, _ = planificador.planificar(L, lo, pos, vpn["xy"], casco=True)
        if wp is None:
            act["saltados"].append(paso)
            msgs.append(f"fina: hallazgo {act['h']['id']} ({NOMBRE_PASO[paso]}): sin ruta hasta ({vpn['xy'][0]:.1f}, {vpn['xy'][1]:.1f}), se salta")
            continue
        if len(wp) == 0:
            wp = np.array([vpn["xy"]])
        ins["vp"] = {"xy": vpn["xy"], "yaw": vpn["yaw"], "esperados": [], "t0": t, "fino": {"paso": paso, "d": vpn["d"]}}
        ins["listo"], ins["t_ok"] = False, None
        msgs.append(f"fina: hallazgo {act['h']['id']} ({act['k']} de {fi['total']}), foto {NOMBRE_PASO[paso]}: ir a ({vpn['xy'][0]:.1f}, {vpn['xy'][1]:.1f}) "
                    f"mirando a {math.degrees(vpn['yaw']):.0f} grados, camara a {vpn['d']:.1f} m de la pared")
        return wp, msgs


def _afinar(fi, dn, act, cam_pos, R_cam, pos_est, R_est, depth, f):
    """Con la foto de ahora (grilla fina) vuelve a ubicar el dano dentro de la pared del hallazgo. Devuelve cuantos recortes positivos cayeron en el hallazgo;
    con 1 o mas, la zona del hallazgo pasa a ser la que abarcan. Con 0 (el detector no lo vio esta vez) queda la zona que habia."""
    alto_img, ancho_img = depth.shape
    cols, filas = fi["grilla"]
    z0 = act["zona"]
    marco = {"n": z0["n"], "t": z0["t"], "p": z0["p"], "s": (z0["s"][0] - MARGEN_UNION, z0["s"][1] + MARGEN_UNION),
             "z": (max(0.0, z0["z"][0] - MARGEN_UNION), min(dn["alto"], z0["z"][1] + MARGEN_UNION))}
    planos = danos._planos(pos_est, R_est, depth, f, dn["alto"], dn["rng"])
    rangos, positivos = [], 0
    for r in range(filas):
        for c in range(cols):
            vent = (c * ancho_img / cols, r * alto_img / filas, (c + 1) * ancho_img / cols, (r + 1) * alto_img / filas)
            if not danos._veredicto(dn, cam_pos, R_cam, depth, f, vent, falsos=dn["falsos"] / (cols * filas))["hay"]:
                continue
            zt = [q for q in (danos._zona(pl, vent, dn["alto"]) for pl in planos) if q is not None]
            if not zt:
                continue
            q = max(zt, key=lambda w: w["k"])
            if danos._se_juntan(marco, q):
                rg = danos._en_marco(z0, q)
                if rg is not None:
                    rangos.append(rg)
                    positivos += 1
    if not rangos:
        return 0, cols * filas
    act["zona"] = {"n": z0["n"], "t": z0["t"], "p": z0["p"], "s": (min(r[0] for r in rangos), max(r[1] for r in rangos)),
                   "z": (min(r[2] for r in rangos), max(r[3] for r in rangos))}
    act["afinada"] = True
    return positivos, cols * filas


def _recorte(img, zona, pos_est, R_est, f, forma_prof):
    """La foto recortada (cuadrada) alrededor de la zona del dano, o la foto entera si el dano no se puede proyectar. El back achica todo a 224 x 224."""
    alto_p, ancho_p = forma_prof
    esc = img.width / ancho_p
    esquinas = []
    for s in zona["s"]:
        xy = zona["p"] + zona["t"] * s
        for z in zona["z"]:
            esquinas.append([xy[0], xy[1], z])
    pc = (np.array(esquinas) - np.asarray(pos_est)) @ np.asarray(R_est)
    fwd = -pc[:, 2]
    if (fwd < 0.2).any():
        return img
    u = (ancho_p / 2 + f * pc[:, 0] / fwd) * esc
    v = (alto_p / 2 - f * pc[:, 1] / fwd) * esc
    cx, cy = 0.5 * (u.min() + u.max()), 0.5 * (v.min() + v.max())
    lado = max(u.max() - u.min(), v.max() - v.min()) * (1 + 2 * MARGEN_RECORTE)
    lado = float(np.clip(lado, LADO_MIN_PX, min(img.width, img.height)))
    x0 = float(np.clip(cx - lado / 2, 0, img.width - lado))
    y0 = float(np.clip(cy - lado / 2, 0, img.height - lado))
    return img.crop((int(round(x0)), int(round(y0)), int(round(x0 + lado)), int(round(y0 + lado))))


def fina_foto(fi, ins, slam, data, cam_id, torso_id, depth, f, img, registro, dn):
    """Llamar justo despues de sacar una foto de inspeccion (y antes de inspeccion_foto_hecha). Si el punto de vista era de la inspeccion fina: afina la ubicacion
    del dano, guarda la foto (recortada) para mandarla cuando termine el hallazgo y devuelve el texto para la consola. Si no, devuelve None."""
    if fi is None or ins["vp"] is None or not ins["vp"].get("fino"):
        return None
    info, act = ins["vp"]["fino"], fi["act"]
    cam_pos, R_cam = data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy()
    R_est = slam["R"] @ (data.xmat[torso_id].reshape(3, 3).T @ R_cam)         # la misma pose estimada que usa dn_foto
    pos_est = np.array([slam["tr"][0], slam["tr"][1], cam_pos[2]])
    if info["paso"] == "general":
        act["d_general"] = info["d"]
    zona_antes, afinada_antes = act["zona"], act["afinada"]
    pos, tot = _afinar(fi, dn, act, cam_pos, R_cam, pos_est, R_est, depth, f)
    if info["paso"] in ("izq", "der") and afinada_antes:
        act["zona"] = zona_antes                       # las fotos de costado confirman, pero no mueven una zona que ya se afino de frente
    act["n_ch"] += 1
    act["conf"] += pos
    detalle = f"zona afinada con {pos} de {tot} recortes" if pos else f"el detector no ve nada en el hallazgo ({tot} recortes)"
    if info["paso"] in ("izq", "der") and pos:
        detalle = f"el detector lo ve en {pos} de {tot} recortes"
    if act["conf"] == 0 and (act["n_ch"] >= FOTOS_PARA_DESCARTAR or not act["pasos"]):
        act["descartado"], act["pasos"] = True, []
        detalle += f"; sin dano a la vista en {act['n_ch']} foto(s): se descarta"
    z = act["zona"]
    c = _centro(z)
    forma = depth.shape
    envio = _recorte(img, z, pos_est, R_est, f, forma) if fi["recorte"] else img
    fi["n_fotos"] += 1
    act["fotos"].append({"paso": info["paso"], "img": envio, "frame": registro["frame"], "timestamp": registro["timestamp"], "d": info["d"], "resp": None})
    vistos = []
    for d_real in dn["reales"]:                                              # solo para validar: el robot no conoce los danos reales
        fr = danos._fraccion_visible(d_real, cam_pos, R_cam, depth, f, (0, 0, forma[1], forma[0]))
        if fr >= danos.FRAC_MIN:
            vistos.append(f"#{d_real['i']} ({100 * fr:.1f}% de la imagen)")
    a_la_vista = ("; reales a la vista: " + ", ".join(vistos)) if vistos else "; ningun dano real a la vista"
    ins["vp"], ins["listo"], ins["t_ok"] = None, False, None                 # libera el punto de vista (el launch no llama a inspeccion_foto_hecha con estas fotos)
    return (f"fina: hallazgo {act['h']['id']}, foto {NOMBRE_PASO[info['paso']]} a {info['d']:.1f} m" + (f" ({detalle})" if detalle else "")
            + f": dano en ({c[0]:.2f}, {c[1]:.2f}), ancho {z['s'][1] - z['s'][0]:.1f} m, altura {z['z'][0]:.1f}-{z['z'][1]:.1f} m{a_la_vista}")


def fina_cierre(fi, dn, slam, ruta_base):
    """Al terminar: resumen para la consola (y un mapa con la zona afinada de cada hallazgo contra los danos reales del yaml)."""
    lineas = []
    reales = danos._real_en_mapa(dn, slam)
    for act in sorted(fi["hechos"], key=lambda a: a["h"]["id"]):
        z = act["zona"]
        c = _centro(z)
        cl = act.get("clase") or (_clase_final(act)[0] if act["fotos"] and not act["descartado"] else None)
        r = next((q for q in reales if danos._coincide(z, q)), None)
        if act["descartado"]:
            r0 = next((q for q in reales if danos._coincide(act["zona0"], q)), None)
            lineas.append(f"[g1_nav] hallazgo {act['h']['id']} descartado: " + (f"ERA un dano real del yaml ({r0['tipo']} #{r0['i']}): se perdio" if r0 else "no habia dano real del yaml (descarte correcto)"))
            continue
        if r is not None:
            ds = float((_centro(z) - r["c"]) @ z["t"])
            dz = 0.5 * (z["z"][0] + z["z"][1]) - r["z"]
            real = f"; en el yaml: {r['tipo']} #{r['i']} (el centro quedo a {abs(ds):.2f} m a lo largo y {abs(dz):.2f} m de alto del real)"
        else:
            real = "; no coincide con ningun dano del yaml"
        clase = f"{cl['label']} {cl['prob']:.0f}%" if cl else "sin clase"
        lineas.append(f"[g1_nav] hallazgo {act['h']['id']} final: ({c[0]:.2f}, {c[1]:.2f}), ancho {z['s'][1] - z['s'][0]:.1f} m, altura {z['z'][0]:.1f}-{z['z'][1]:.1f} m, "
                      f"{len(act['fotos'])} fotos{' (saltadas: ' + ', '.join(act['saltados']) + ')' if act['saltados'] else ''}, {clase}{real}")
    lat = fi["lat"]
    desc = sum(1 for a in fi["hechos"] if a["descartado"])
    lineas.append(f"[g1_nav] inspeccion fina: {len(fi['hechos'])} de {fi['total']} hallazgos ({desc} descartados), {fi['n_fotos']} fotos; el back contesto {len(lat)}"
                  + (f" (en promedio {sum(lat) / len(lat):.2f} s, maximo {max(lat):.2f} s)" if lat else "") + f"; con error {fi['errores']}; hallazgos sin clase: {fi['sin_clase']}")
    if any(all(fo.get("resp") is None for fo in a["fotos"]) for a in fi["hechos"] if a["fotos"] and not a["descartado"]) or fi["errores"]:
        lineas.append("[g1_nav] (si el back no contesto o no clasifico: ¿esta corriendo uvicorn? Si contesto pero sin clases, es que su detector no marco dano en esas fotos y entonces el back no clasifica)")
    try:
        hall = [{"n": a["zona"]["n"], "t": a["zona"]["t"], "p": a["zona"]["p"], "s": a["zona"]["s"], "z": a["zona"]["z"], "id": a["h"]["id"]} for a in fi["hechos"] if not a["descartado"]]
        dn2 = {"reales": dn["reales"], "dets": [], "hallazgos": hall}
        L, lo = slam_recortar(slam)
        danos.dn_dibujar(dn2, slam, L, lo, ruta_base + "_fina.png")
        lineas.append(f"[g1_nav] mapa de la inspeccion fina: {ruta_base}_fina.png (rojo = dano real del yaml, verde = zona afinada que coincide con uno, naranja = sin dano real)")
    except Exception as e:   # el dibujo es accesorio
        lineas.append(f"[g1_nav] no se pudo dibujar el mapa de la inspeccion fina ({e})")
    return "\n".join(lineas)
