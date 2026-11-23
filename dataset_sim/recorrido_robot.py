"""
Simula al G1 recorriendo una casa y guarda lo que ve la camara de la cabeza.

El recorrido imita una inspeccion: el robot entra a cada sala, se para en el
medio y gira una vuelta completa, despues se acerca a cada pared (a 1,1 m),
la mira de frente y sigue hasta la proxima. Entre un punto y otro camina por
el medio del espacio libre, pasando por las puertas. Mientras camina se saca
una foto cada 0,2 m; cuando gira, una cada 15 grados.

Cada foto se etiqueta igual que en generar_capturas.py (damaged si el daño
ocupa al menos el 2 % de la foto, undamaged si no hay daño a la vista y
descartar si hay un poco). Si el escenario trae danos_piso.json, tambien
cuenta el daño del piso.

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/recorrido_robot.py --escenario variantes/casa_blanca --salida dataset_sim/recorrido_casa_blanca

Salida:
    <salida>/<escenario>/frames/NNNN.png
    <salida>/capturas.csv   mismo formato que generar_capturas.py, mas paso, accion y sala
    <salida>/plano.png      planta con el recorrido
"""
import argparse
import csv
import heapq
import json
import math
import os

import mujoco
import numpy as np
import yaml
from matplotlib.path import Path
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter

import generar_capturas as gc

RES = 0.05                 # metros por celda del plano
RADIO_ROBOT = 0.28         # distancia minima del centro del robot a paredes y muebles (el G1 mide unos 0,45 m de ancho)
DIST_PARED = 1.1           # distancia a la que se para para mirar una pared
PASO = 0.2                 # metros entre fotos mientras camina
GIRO = math.radians(15)    # grados entre fotos mientras gira


def plano(modelo, datos, nombres, interior):
    todos = np.array([p for pol in interior for p in pol])
    x0, y0 = todos.min(axis=0) - 0.5
    x1, y1 = todos.max(axis=0) + 0.5
    nx, ny = int((x1 - x0) / RES) + 1, int((y1 - y0) / RES) + 1
    xs, ys = x0 + (np.arange(nx) + 0.5) * RES, y0 + (np.arange(ny) + 0.5) * RES
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    puntos = np.stack([gx.ravel(), gy.ravel()], axis=1)
    dentro = np.zeros(nx * ny, bool)
    sala = np.full(nx * ny, -1)
    for k, pol in enumerate(interior):
        en = Path(pol).contains_points(puntos)
        dentro |= en
        sala[en & (sala < 0)] = k
    dentro = binary_dilation(dentro.reshape(nx, ny), iterations=2)
    ocupado = np.zeros((nx, ny), bool)
    for i, n in enumerate(nombres):
        if not n.startswith("col_") or modelo.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        medio = np.abs(datos.geom_xmat[i].reshape(3, 3)) @ modelo.geom_size[i]
        c = datos.geom_xpos[i]
        if c[2] + medio[2] < 0.05 or c[2] - medio[2] > 1.4:
            continue
        a0, a1 = int((c[0] - medio[0] - x0) / RES), int(math.ceil((c[0] + medio[0] - x0) / RES))
        b0, b1 = int((c[1] - medio[1] - y0) / RES), int(math.ceil((c[1] + medio[1] - y0) / RES))
        ocupado[max(0, a0):max(0, a1), max(0, b0):max(0, b1)] = True
    libre = dentro & ~ocupado
    return dict(x0=x0, y0=y0, nx=nx, ny=ny, libre=libre, dist=distance_transform_edt(libre) * RES,
                sala=sala.reshape(nx, ny), ocupado=ocupado)


def celda(p, x, y):
    return int((x - p["x0"]) / RES), int((y - p["y0"]) / RES)


def punto(p, i, j):
    return p["x0"] + (i + 0.5) * RES, p["y0"] + (j + 0.5) * RES


def camino(p, a, b):
    """A* sobre el plano; prefiere ir por el medio del espacio libre."""
    ok = p["dist"] >= RADIO_ROBOT
    costo_celda = 1.0 + 2.0 * np.clip(0.8 - p["dist"], 0, None)
    vecinos = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
    abiertos = [(0.0, 0.0, a)]
    previo, costo = {a: None}, {a: 0.0}
    while abiertos:
        _, g, c = heapq.heappop(abiertos)
        if c == b:
            break
        if g > costo[c]:
            continue
        for di, dj, largo in vecinos:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < p["nx"] and 0 <= n[1] < p["ny"]) or not ok[n]:
                continue
            ng = g + largo * costo_celda[n]
            if ng < costo.get(n, 1e18):
                costo[n], previo[n] = ng, c
                heapq.heappush(abiertos, (ng + math.hypot(b[0] - n[0], b[1] - n[1]), ng, n))
    if b not in previo:
        return None
    ruta, c = [], b
    while c is not None:
        ruta.append(punto(p, *c))
        c = previo[c]
    return np.array(ruta[::-1])


def suavizar(ruta, veces=3):
    for _ in range(veces):
        if len(ruta) < 3:
            break
        r = ruta.copy()
        r[1:-1] = (ruta[:-2] + 2 * ruta[1:-1] + ruta[2:]) / 4
        ruta = r
    return ruta


def remuestrear(ruta, paso):
    tramos = np.hypot(*np.diff(ruta, axis=0).T)
    s = np.concatenate([[0], np.cumsum(tramos)])
    if s[-1] < 1e-6:
        return ruta[:1], s[-1]
    t = np.arange(0, s[-1], paso)
    return np.stack([np.interp(t, s, ruta[:, 0]), np.interp(t, s, ruta[:, 1])], axis=1), s[-1]


def libre_cerca(p, x, y, minimo):
    i, j = celda(p, x, y)
    if 0 <= i < p["nx"] and 0 <= j < p["ny"] and p["dist"][i, j] >= minimo:
        return i, j
    return None


def objetivos_de_sala(p, k, pol):
    """Centro de la sala (el punto mas lejos de todo) y un punto frente a cada pared."""
    en_sala = (p["sala"] == k) & (p["dist"] >= RADIO_ROBOT)
    if not en_sala.any():
        return None, []
    d = np.where(en_sala, p["dist"], -1)
    centro = np.unravel_index(np.argmax(d), d.shape)
    pol = np.array(pol)
    area = 0.5 * np.sum(pol[:, 0] * np.roll(pol[:, 1], -1) - np.roll(pol[:, 0], -1) * pol[:, 1])
    paredes = []
    for a, b in zip(pol, np.roll(pol, -1, axis=0)):
        largo = np.hypot(*(b - a))
        if largo < 1.0:
            continue
        u = (b - a) / largo
        n = np.array([-u[1], u[0]]) * (1 if area > 0 else -1)     # normal hacia adentro de la sala
        hallado = None
        for dist in (DIST_PARED, 1.3, 1.6):
            for corr in (0.0, -0.4, 0.4, -0.8, 0.8):
                if abs(corr) > largo / 2 - 0.3:
                    continue
                q = (a + b) / 2 + u * corr + n * dist
                c = libre_cerca(p, q[0], q[1], RADIO_ROBOT + 0.05)
                if c is not None and p["sala"][c] == k:
                    hallado = (c, math.atan2(-n[1], -n[0]))
                    break
            if hallado:
                break
        if hallado:
            paredes.append(hallado)
    return centro, paredes


def armar_recorrido(p, interior, inicio):
    """Lista de poses (x, y, yaw, accion, sala) del recorrido completo."""
    pendientes = list(range(len(interior)))
    poses = []
    actual, yaw = inicio, 0.0
    s_total = 0.0

    def ir(destino):
        nonlocal actual, yaw, s_total
        if destino == actual:
            return True
        r = camino(p, actual, destino)
        if r is None or len(r) < 2:
            return False
        pts, largo = remuestrear(suavizar(r), PASO)
        for idx in range(len(pts)):
            adelante = pts[min(idx + 3, len(pts) - 1)] - pts[idx]
            if np.hypot(*adelante) > 0.05:
                if idx == 0:
                    girar_hacia(math.atan2(adelante[1], adelante[0]), pts[idx])
                yaw = math.atan2(adelante[1], adelante[0])
            s_total += PASO
            poses.append((pts[idx][0], pts[idx][1], yaw, "caminando", s_total))
        actual = destino
        return True

    def girar_hacia(objetivo, pos, vuelta=False):
        nonlocal yaw
        delta = (objetivo - yaw + math.pi) % (2 * math.pi) - math.pi
        if vuelta:
            delta = 2 * math.pi
        n = int(abs(delta) / GIRO)
        for t in range(1, n + 1):
            poses.append((pos[0], pos[1], yaw + delta * t / n, "girando" if not vuelta else "vuelta", s_total))
        yaw = (yaw + delta + math.pi) % (2 * math.pi) - math.pi

    while pendientes:
        ax, ay = punto(p, *actual)
        orden = []
        for k in pendientes:
            centro, paredes = objetivos_de_sala(p, k, interior[k])
            if centro is None:
                continue
            cx, cy = punto(p, *centro)
            orden.append((math.hypot(cx - ax, cy - ay), k, centro, paredes))
        if not orden:
            break
        orden.sort(key=lambda o: o[0])
        _, k, centro, paredes = orden[0]
        pendientes.remove(k)
        if not ir(centro):
            continue
        girar_hacia(yaw, punto(p, *centro), vuelta=True)
        for c, mira in paredes:
            if ir(c):
                girar_hacia(mira, punto(p, *c))
                x, y = punto(p, *c)
                for extra in (-0.35, 0.0, 0.35, 0.0):
                    poses.append((x, y, mira + extra, "mirando_pared", s_total))
                yaw = mira
    return poses


def cargar_danos_piso(carpeta, nombres):
    """Daños de piso (danos_piso.json, si el escenario los tiene) con su mascara en metros."""
    ruta = os.path.join(carpeta, "danos_piso.json")
    if not os.path.exists(ruta):
        return {}
    with open(ruta, encoding="utf-8") as f:
        lista = json.load(f)
    pisos = {}
    for d in lista:
        gid = nombres.index(f"vis_{d['nombre']}")
        m = np.array(Image.open(os.path.join(carpeta, "assets", f"{d['nombre']}_mascara.png"))) > 0
        pisos[gid] = dict(d, mascara=maximum_filter(m, size=13))
    return pisos


def etiquetar_piso(modelo, datos, render, cam_id, pisos):
    """Fraccion de la foto con daño de piso y tipos vistos."""
    if not pisos:
        return 0.0, set()
    render.enable_segmentation_rendering()
    render.update_scene(datos, camera=gc.CAMARA)
    seg = render.render()[..., 0].copy()
    render.disable_segmentation_rendering()
    if not np.isin(seg, list(pisos)).any():
        return 0.0, set()
    render.enable_depth_rendering()
    render.update_scene(datos, camera=gc.CAMARA)
    prof = render.render().copy()
    render.disable_depth_rendering()
    f = (gc.ALTO / 2) / math.tan(math.radians(modelo.cam_fovy[cam_id]) / 2)
    u, v = np.meshgrid(np.arange(gc.ANCHO) + 0.5, np.arange(gc.ALTO) + 0.5)
    rayos = np.stack([(u - gc.ANCHO / 2) / f, -(v - gc.ALTO / 2) / f, -np.ones_like(u)], axis=-1) * prof[..., None]
    mundo = rayos @ datos.cam_xmat[cam_id].reshape(3, 3).T + datos.cam_xpos[cam_id]
    total, tipos = np.zeros(seg.shape, bool), set()
    for gid, d in pisos.items():
        en = seg == gid
        if not en.any():
            continue
        m = d["mascara"]
        H, W = m.shape
        c = ((mundo[..., 0] - d["x0"]) / (d["x1"] - d["x0"]) * W).astype(int).clip(0, W - 1)
        r = ((d["y1"] - mundo[..., 1]) / (d["y1"] - d["y0"]) * H).astype(int).clip(0, H - 1)
        dentro = en & m[r, c]
        if dentro.sum() > 0.003 * seg.size:
            tipos.add(d["tipo"] + "_piso")
        total |= dentro
    return total.sum() / seg.size, tipos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenario", default="variantes/casa_blanca")
    ap.add_argument("--salida", default=os.path.join(gc.RAIZ, "dataset_sim", "recorrido_casa_blanca"))
    args = ap.parse_args()

    modelo, datos = gc.construir_modelo(args.escenario)
    nombres = gc.nombres_geoms(modelo)
    cam_id = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_CAMERA, gc.CAMARA)
    et = args.escenario.split("/")[-1]
    with open(os.path.join(gc.RAIZ, "g1_navigation_integration", "policy", "deploy.yaml"), encoding="utf-8") as f:
        articulaciones = np.array(yaml.safe_load(f)["default_joint_pos"], dtype=float)
    postura = datos.qpos.copy()
    adr = modelo.joint("floating_base_joint").qposadr[0]
    postura[adr + 7:adr + 7 + len(articulaciones)] = articulaciones
    gc.colocar_robot(modelo, datos, 0, 0, 0, 0, 100, postura)
    danos = gc.cargar_danos(modelo, datos, nombres, args.escenario)
    _, brechas = gc.limites_y_brechas(modelo, datos, nombres, args.escenario)
    carpeta = os.path.join(gc.RAIZ, "escenarios", args.escenario)
    with open(os.path.join(carpeta, "interior.json"), encoding="utf-8") as f:
        interior = json.load(f)
    with open(os.path.join(carpeta, "variante.json"), encoding="utf-8") as f:
        tipos_sala = json.load(f).get("salas", [])
    pisos = cargar_danos_piso(carpeta, nombres)

    p = plano(modelo, datos, nombres, interior)
    inicio = libre_cerca(p, 0.0, 0.0, RADIO_ROBOT)
    if inicio is None:
        libres = np.argwhere(p["dist"] >= RADIO_ROBOT)
        inicio = tuple(libres[np.argmin(np.hypot(*(libres * RES + [p["x0"], p["y0"]]).T))])
    poses = armar_recorrido(p, interior, inicio)
    print(f"{et}: {len(poses)} fotos en el recorrido", flush=True)

    os.makedirs(os.path.join(args.salida, et, "frames"), exist_ok=True)
    render = mujoco.Renderer(modelo, gc.ALTO, gc.ANCHO)
    filas = []
    for n, (x, y, yaw, accion, s) in enumerate(poses):
        # balanceo de la cabeza al caminar: un paso cada 0,3 m
        fase = 2 * math.pi * s / 0.6
        dz = 0.012 * math.sin(2 * fase) if accion == "caminando" else 0.0
        cab = math.radians(1.2) * math.sin(fase) if accion == "caminando" else 0.0
        gc.colocar_robot(modelo, datos, x, y, yaw, cab, dz, postura)
        clase, f_pared, f_obj, tipos = gc.etiquetar(modelo, datos, render, cam_id, nombres, danos, brechas)
        f_piso, tipos_piso = etiquetar_piso(modelo, datos, render, cam_id, pisos)
        if f_piso > 0:
            tipos = sorted(set(tipos) | tipos_piso)
            if f_pared + f_piso >= gc.MIN_DANO:
                clase = "damaged"
            elif clase == "undamaged":
                clase = "descartar"
        render.update_scene(datos, camera=gc.CAMARA)
        nombre = f"{n:04d}.png"
        Image.fromarray(render.render()).save(os.path.join(args.salida, et, "frames", nombre))
        i, j = celda(p, x, y)
        k = p["sala"][min(max(i, 0), p["nx"] - 1), min(max(j, 0), p["ny"] - 1)]
        filas.append(dict(archivo=f"{et}/frames/{nombre}", escenario=et, clase=clase, variante=0, x=round(x, 3),
                          y=round(y, 3), yaw=round(yaw, 3), dano_pared=round(f_pared, 4), dano_piso=round(f_piso, 4), otros_danos=round(f_obj, 4),
                          tipos="|".join(tipos), paso=n, accion=accion, metros=round(s, 2),
                          sala=tipos_sala[k] if 0 <= k < len(tipos_sala) else ""))
    render.close()
    with open(os.path.join(args.salida, "capturas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    escala = 4
    img = Image.new("RGB", (p["nx"] * escala, p["ny"] * escala), (255, 255, 255))
    pix = np.full((p["ny"], p["nx"], 3), 255, np.uint8)
    pix[p["ocupado"].T] = (90, 90, 90)
    pix[(~p["ocupado"] & (p["sala"] < 0)).T] = (225, 225, 225)
    img = Image.fromarray(pix[::-1]).resize((p["nx"] * escala, p["ny"] * escala), Image.NEAREST)
    d = ImageDraw.Draw(img)
    xy = [((f["x"] - p["x0"]) / RES * escala, (p["ny"] - (f["y"] - p["y0"]) / RES) * escala) for f in filas]
    d.line(xy, fill=(47, 109, 176), width=3)
    for f, (u, v) in zip(filas, xy):
        if f["clase"] == "damaged":
            d.ellipse([u - 4, v - 4, u + 4, v + 4], fill=(200, 40, 40))
    img.save(os.path.join(args.salida, "plano.png"))
    cuenta = {c: sum(f["clase"] == c for f in filas) for c in ("undamaged", "damaged", "descartar")}
    print(f"listo: {len(filas)} fotos {cuenta}, {filas[-1]['metros']:.0f} m caminados")


if __name__ == "__main__":
    main()
