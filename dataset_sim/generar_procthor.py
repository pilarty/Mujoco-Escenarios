"""
Convierte casas de ProcTHOR (allenai, conjunto procthor-10k) en escenarios de MuJoCo.

De cada casa se toma la planta: salas, paredes, puertas, ventanas y muebles
(como cajas del tamaño real de cada mueble, segun la base de assets de ProcTHOR).
Las texturas de paredes, pisos y muebles y los daños de pared son los mismos de
generar_realista.py (fotos reales). La casa se corre para que el origen (0, 0)
quede libre dentro de una sala, donde arranca el robot.

Datos (en dataset_sim/procthor/):
    casas_60.json         primeras 60 casas de procthor-10k (split train)
    asset-database.json   medidas de los muebles (repositorio allenai/procthor)

Salida: escenarios/variantes/procthor_NN/ con escenario.xml, meshes/, assets/,
damage_manifest.json, variante.json e interior.json (poligonos de las salas).

Con --estilo realista las paredes son de revoque liso pintado, los daños
salen de danos_realistas.py (grietas con bordes astillados, humedad que sube
del piso o baja del techo, revoque caido) y los muebles tapizados llevan tela.
Ademas, detalles_casa.py arma los muebles con forma (patas, respaldos,
cajones, libros), agrega los objetos que ProcTHOR pone sobre los muebles y en
las paredes (televisor, cuadros, espejos, lamparas, plantas), zocalos,
alfeizares, llaves de luz, tomacorrientes, lamparas de techo, sombras debajo
de los muebles y piso de ceramica en baños y cocinas.
Esa version se guarda en escenarios/variantes/procthor_real_NN/, sin pisar
las casas procthor_NN.

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/generar_procthor.py --casas 3 8 10 20 28
    python dataset_sim/generar_procthor.py --casas 3 8 10 20 28 --estilo realista
"""
import glob
import argparse
import json
import math
import re
import os
import shutil

import numpy as np

import danos_realistas as dr
import detalles_casa as dc
import generar_realista as gr
import generar_variantes as gv

DATOS = os.path.join(gv.RAIZ, "dataset_sim", "procthor")
OMITIR = {"Painting", "Window", "Doorway", "Doorframe", "LightSwitch", "Mirror", "Curtains", "Blinds", "Poster",
          "ShowerCurtain", "Television", "Towel", "TowelHolder", "HandTowel", "HandTowelHolder", "ToiletPaperHanger"}
COLORES = {
    "Bed": (0.86, 0.85, 0.82), "Sofa": (0.42, 0.42, 0.45), "ArmChair": (0.45, 0.38, 0.32),
    "Fridge": (0.88, 0.88, 0.88), "Toilet": (0.95, 0.95, 0.95), "Bathtub": (0.95, 0.95, 0.95),
    "Sink": (0.92, 0.92, 0.92), "StoveBurner": (0.2, 0.2, 0.2), "Microwave": (0.25, 0.25, 0.25),
    "GarbageCan": (0.3, 0.3, 0.32), "Chair": (0.35, 0.25, 0.18), "Ottoman": (0.5, 0.4, 0.3),
    "DogBed": (0.55, 0.45, 0.35), "LaundryHamper": (0.7, 0.65, 0.55), "Safe": (0.3, 0.3, 0.3),
}
MADERA = {"DiningTable", "CoffeeTable", "SideTable", "Desk", "Dresser", "Shelf", "ShelvingUnit", "Bookcase",
          "TVStand", "CounterTop", "Countertop", "Cabinet", "Drawer", "Stool"}


def punto_en(pol, x, y):
    dentro = False
    n = len(pol)
    for i in range(n):
        (x1, y1), (x2, y2) = pol[i], pol[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            dentro = not dentro
    return dentro


def unir(intervalos):
    out = []
    for a, b in sorted(intervalos):
        if out and a <= out[-1][1] + 0.02:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def cargar_casa(indice):
    with open(os.path.join(DATOS, "casas_60.json"), encoding="utf-8") as f:
        return json.load(f)[indice]


def cargar_assets():
    with open(os.path.join(DATOS, "asset-database.json"), encoding="utf-8") as f:
        db = json.load(f)
    return {a["assetId"]: a for lista in db.values() for a in lista}


def muebles_de(casa, assets):
    """Cajas (x0, x1, y0, y1, z0, z1, tipo) de los muebles que apoyan en el piso, en coordenadas ProcTHOR."""
    cajas = []
    for o in casa["objects"]:
        a = assets.get(o["assetId"])
        if not a or a["objectType"] in OMITIR:
            continue
        bb = a["boundingBox"]
        lx, ly, lz = bb["x"], bb["z"], bb["y"]
        rot = round(o["rotation"]["y"]) % 180
        if 45 <= rot < 135:
            lx, ly = ly, lx
        z0 = max(0.0, o["position"]["y"] - lz / 2)
        if a["objectType"] == "Television" or z0 > 0.35 or max(lx, ly) < 0.25 or lz < 0.2:
            continue
        cx, cy = o["position"]["x"], o["position"]["z"]
        cajas.append((cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, z0, z0 + lz, a["objectType"],
                      frente_de(o["rotation"]["y"])))
    return cajas


def frente_de(rot):
    """Hacia donde mira el frente del objeto: en ProcTHOR es +z local, (sin r, cos r) en x, y."""
    return {0: "+y", 1: "+x", 2: "-y", 3: "-x"}[int(round((rot % 360) / 90)) % 4]


COLGADOS = {"Painting", "Mirror", "Television", "Poster"}


def objetos_extra(casa, assets):
    """Objetos sobre los muebles (hijos) y colgados en la pared: (x0, x1, y0, y1, z0, z1, tipo, frente, clase)."""
    out = []

    def caja_de(o, a):
        bb = a["boundingBox"]
        lx, ly, lz = bb["x"], bb["z"], bb["y"]
        if 45 <= round(o["rotation"]["y"]) % 180 < 135:
            lx, ly = ly, lx
        cx, cy, cz = o["position"]["x"], o["position"]["z"], o["position"]["y"]
        return cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, cz - lz / 2, cz + lz / 2

    def visitar(o, nivel):
        a = assets.get(o["assetId"])
        if a:
            t = a["objectType"]
            if nivel > 0 and t not in ("Faucet", "ToiletPaper", "Pen", "Pencil", "CreditCard", "KeyChain",
                                       "Spoon", "Fork", "Knife", "ButterKnife", "CD", "Egg", "Watch"):
                c = caja_de(o, a)
                if max(c[1] - c[0], c[3] - c[2], c[5] - c[4]) >= 0.07:
                    out.append((*c, t, frente_de(o["rotation"]["y"]), "hijo"))
            elif nivel == 0 and t in COLGADOS:
                out.append((*caja_de(o, a), t, frente_de(o["rotation"]["y"]), "pared"))
        for h in o.get("children") or []:
            visitar(h, nivel + 1)

    for o in casa["objects"]:
        visitar(o, 0)
    return out


def elegir_origen(salas, cajas):
    """Punto libre dentro de la sala mas grande, lo mas lejos posible de paredes y muebles."""
    mejor, dist = None, -1.0
    for pol in sorted(salas.values(), key=lambda p: -area(p))[:2]:
        xs, ys = [p[0] for p in pol], [p[1] for p in pol]
        for x in np.arange(min(xs) + 0.5, max(xs) - 0.49, 0.1):
            for y in np.arange(min(ys) + 0.5, max(ys) - 0.49, 0.1):
                if not punto_en(pol, x, y):
                    continue
                d = min(dist_poligono(pol, x, y), *(dist_caja(c, x, y) for c in cajas)) if cajas else dist_poligono(pol, x, y)
                if d > dist:
                    mejor, dist = (float(x), float(y)), d
    return mejor, dist


def area(pol):
    return abs(sum(pol[i][0] * pol[(i + 1) % len(pol)][1] - pol[(i + 1) % len(pol)][0] * pol[i][1]
                   for i in range(len(pol)))) / 2


def dist_poligono(pol, x, y):
    d = 1e9
    for i in range(len(pol)):
        (x1, y1), (x2, y2) = pol[i], pol[(i + 1) % len(pol)]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / max(1e-9, dx * dx + dy * dy)))
        d = min(d, math.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return d


def dist_caja(c, x, y):
    return math.hypot(max(c[0] - x, 0, x - c[1]), max(c[2] - y, 0, y - c[3]))


TAPIZADOS = {"Bed", "Sofa", "ArmChair", "Ottoman", "DogBed", "LaundryHamper"}


def generar(indice, numero, estilo="original"):
    realista = estilo == "realista"
    nombre = f"procthor_real_{numero:02d}" if realista else f"procthor_{numero:02d}"
    carpeta = os.path.join(gv.DESTINO, nombre)
    if os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
    os.makedirs(os.path.join(carpeta, "assets"))
    os.makedirs(os.path.join(carpeta, "meshes"))
    rng = np.random.default_rng(3000 + numero)
    gv.PROB_PINTAR = 0.0
    gv.OVERLAY_REAL = gr.overlay_real
    gr._assets[0] = os.path.join(carpeta, "assets")
    texturas = gv.preparar_texturas(rng, os.path.join(carpeta, "assets"))
    if realista:
        dr.ASSETS[0] = os.path.join(carpeta, "assets")
        claras = sorted(r for r in glob.glob(os.path.join(gv.TEXTURAS, "pared__*.jpg"))
                        if any(c in r for c in gv.PAREDES_CLARAS))
        base = claras[rng.integers(len(claras))]
        color = dr.pintar_pared(base, os.path.join(carpeta, "assets", "pared_revoque.png"), rng)
        texturas["pared"] = f"{os.path.basename(base)} lisa pintada {color}"
        dr.textura_tela(os.path.join(carpeta, "assets", "tela.png"), rng)
        maderas = [os.path.join(gv.TEXTURAS, f"piso__{n}.jpg") for n in
                   ("herringbone_parquet", "laminate_floor", "laminate_floor_02", "plank_flooring", "wood_floor")]
        piso = maderas[rng.integers(len(maderas))]
        gv.abrir_textura(piso).save(os.path.join(carpeta, "assets", "piso_madera.png"))
        texturas["pisos"] = [os.path.basename(piso), "ceramica"]

    casa = cargar_casa(indice)
    assets = cargar_assets()
    salas_u = {r["id"]: [(p["x"], p["z"]) for p in r["floorPolygon"]] for r in casa["rooms"]}
    tipo_sala = {r["id"]: r["roomType"] for r in casa["rooms"]}
    cajas_u = muebles_de(casa, assets)
    extras_u = objetos_extra(casa, assets) if realista else []
    origen, holgura = elegir_origen(salas_u, cajas_u)
    ox, oy = origen

    def m(x, y):
        return x - ox, y - oy

    salas = {k: [m(*p) for p in pol] for k, pol in salas_u.items()}
    cajas = [(c[0] - ox, c[1] - ox, c[2] - oy, c[3] - oy, c[4], c[5], c[6], c[7]) for c in cajas_u]
    extras = [(c[0] - ox, c[1] - ox, c[2] - oy, c[3] - oy, *c[4:]) for c in extras_u]

    def sala_de(x, y):
        for k, pol in salas.items():
            if punto_en(pol, x, y):
                return k
        return None

    mod = gv.cargar_generador(nombre)
    mod.AQUI, mod.ASSETS, mod.MESHES = carpeta, os.path.join(carpeta, "assets"), os.path.join(carpeta, "meshes")
    mod.PREF = f"../../escenarios/variantes/{nombre}"
    mod.RNG = np.random.default_rng(4000 + numero)
    T, H = mod.T, mod.H
    todos = [p for pol in salas.values() for p in pol]
    mod.X0, mod.X1 = min(p[0] for p in todos), max(p[0] for p in todos)
    mod.Y0, mod.Y1 = min(p[1] for p in todos), max(p[1] for p in todos)
    mod.TECHO, mod.CAMARAS, mod.PX_PIEZA = True, [], 256

    # ---- lineas de pared: (eje, pos) -> intervalos unidos
    poligono_pared = {}
    lineas, exterior = {}, set()
    for w in casa["walls"]:
        p0, p1 = w["polygon"][0], w["polygon"][1]
        (x0, y0), (x1, y1) = m(p0["x"], p0["z"]), m(p1["x"], p1["z"])
        poligono_pared[w["id"]] = ((x0, y0), (x1, y1))
        if abs(x0 - x1) < 1e-3:
            clave, a, b = ("y", round(x0, 3)), min(y0, y1), max(y0, y1)
        else:
            clave, a, b = ("x", round(y0, 3)), min(x0, x1), max(x0, x1)
        lineas.setdefault(clave, []).append((a, b))
        if "exterior" in w["id"]:
            exterior.add(clave)
    lineas = {k: unir(v) for k, v in lineas.items()}

    # ---- puertas y ventanas sobre las lineas
    aberturas = {}
    centros_puerta = []
    for lista, tipo in ((casa["doors"], "puerta"), (casa["windows"], "ventana")):
        for d in lista:
            (x0, y0), (x1, y1) = poligono_pared[d["wall0"]]
            largo = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / largo, (y1 - y0) / largo
            h0, h1 = d["holePolygon"][0], d["holePolygon"][1]
            c = (h0["x"] + h1["x"]) / 2
            ancho = abs(h1["x"] - h0["x"])
            px, py = x0 + ux * c, y0 + uy * c
            if abs(x0 - x1) < 1e-3:
                clave, s = ("y", round(x0, 3)), py
            else:
                clave, s = ("x", round(y0, 3)), px
            if tipo == "puerta":
                z0, z1 = 0.0, min(h1["y"], H - 0.35)
                centros_puerta.append((px, py))
            else:
                z0, z1 = max(0.3, h0["y"]), min(h1["y"], H - 0.25)
                if z1 - z0 < 0.4:
                    continue
            ab = aberturas.setdefault(clave, [])
            if any(abs(s - o[0]) < (ancho + o[1]) / 2 + 0.05 for o in ab):
                continue
            ab.append((round(s, 3), round(ancho, 3), z0, z1, tipo))

    # ---- daños de pared, solo en caras que dan a una sala y sin muebles delante
    paredes = []
    for (eje, pos), intervalos in lineas.items():
        for i, (a, b) in enumerate(intervalos):
            ab = sorted(o for o in aberturas.get((eje, pos), []) if a < o[0] < b)
            prefijo = "ext" if (eje, pos) in exterior else "int"
            paredes.append(dict(nombre=f"{prefijo}_{eje}_{len(paredes)}", eje=eje, pos=pos, a0=a - T / 2, a1=b + T / 2,
                                ab=ab, danos=[]))
    cruces = {"x": sorted({p for (e, p) in lineas if e == "y"}), "y": sorted({p for (e, p) in lineas if e == "x"})}
    largo_total = sum(p["a1"] - p["a0"] for p in paredes)
    cantidad = int(np.clip(round(largo_total / 9), 4, 10))
    for _ in range(cantidad * 15):
        if sum(len(p["danos"]) for p in paredes) >= cantidad:
            break
        pesos = np.array([p["a1"] - p["a0"] for p in paredes])
        p = paredes[rng.choice(len(paredes), p=pesos / pesos.sum())]
        tipo = str(rng.choice(["grieta_estructural", "humedad", "desprendimiento_revoque"], p=[0.5, 0.25, 0.25]))
        if tipo == "grieta_estructural":
            w, h = rng.uniform(0.8, 1.6), rng.uniform(1.2, 2.2)
        elif tipo == "humedad":
            w, h = rng.uniform(0.9, 1.8), rng.uniform(0.9, 1.8)
        else:
            w, h = rng.uniform(0.6, 1.3), rng.uniform(0.6, 1.2)
        ocupado = list(p["ab"]) + [(c, 0.4) for c in cruces[p["eje"]] if p["a0"] < c < p["a1"]]
        libres = [(a, b) for a, b in gv.tramos_libres(p["a0"], p["a1"], ocupado) if b - a >= w + 0.2]
        if not libres:
            continue
        a, b = libres[rng.integers(len(libres))]
        s = float(rng.uniform(a + w / 2, b - w / 2))
        if not gv.lejos(s, [d["s"] for d in p["danos"]], 2.0):
            continue
        z = float(rng.uniform(h / 2 + 0.1, H - h / 2 - 0.05)) if tipo != "grieta_estructural" else H - h / 2 - 0.02
        desde_abajo = bool(rng.random() < 0.65)
        if realista and tipo == "humedad":
            z = h / 2 + 0.02 if desde_abajo else H - h / 2 - 0.02
        lados = []
        normal = "y" if p["eje"] == "x" else "x"
        for signo, cara in ((-1, normal), (1, normal.upper())):
            off = p["pos"] + signo * (T / 2 + 0.4)
            px, py = (s, off) if p["eje"] == "x" else (off, s)
            sala = sala_de(px, py)
            if sala is None:
                continue
            tapado = any(dist_caja(c, px, py) < w / 2 + 0.3 and c[5] > 0.9 for c in cajas)
            tapado = tapado or any(c[8] == "hijo" and c[5] > 0.9 and dist_caja(c, px, py) < w / 2 + 0.3 for c in extras)
            if not tapado:
                lados.append((cara, sala))
        if not lados:
            continue
        cara, sala = lados[rng.integers(len(lados))]
        p["danos"].append(dict(cara=cara, s=s, z=z, w=float(w), h=float(h), ov=(dr.overlay(tipo, float(w), float(h), rng, desde_abajo) if realista
                                   else gv.OVERLAY_REAL(tipo, float(w), float(h), rng)),
                               tipo=tipo, sala=tipo_sala[sala]))
    for p in paredes:
        mod.pared(p["nombre"], p["eje"], p["pos"], p["a0"], p["a1"], p["ab"], mat="revoque", danos=p["danos"])
    if realista:
        det = dc.Detalles(mod.EXTRA_XML, os.path.join(carpeta, "assets"), mod.PREF, np.random.default_rng(5000 + numero))
        for p in paredes:
            for d in p["danos"]:
                lo, hi = p["pos"] - T / 2 - 0.25, p["pos"] + T / 2 + 0.25
                s0, s1, z0, z1 = d["s"] - d["w"] / 2, d["s"] + d["w"] / 2, d["z"] - d["h"] / 2, d["z"] + d["h"] / 2
                det.zonas_dano.append((s0, s1, lo, hi, z0, z1) if p["eje"] == "x" else (lo, hi, s0, s1, z0, z1))
        for p in paredes:
            det.zocalos(p["eje"], p["pos"], p["a0"], p["a1"], p["ab"], T, mod.SLAB)
            for c, w, z0, z1, tipo in p["ab"]:
                if tipo == "ventana":
                    det.alfeizar(p["eje"], p["pos"], c, w, z0, T)
                elif tipo == "puerta":
                    for lado in (-1, 1):
                        s = c + rng.choice([-1, 1]) * (w / 2 + 0.17)
                        if p["a0"] + 0.1 < s < p["a1"] - 0.1:
                            det.placa(p["eje"], p["pos"], s, 1.1, lado, T)
            for _ in range(int(rng.integers(0, 3))):
                s = float(rng.uniform(p["a0"] + 0.3, p["a1"] - 0.3))
                if all(abs(s - o[0]) > o[1] / 2 + 0.2 for o in p["ab"]):
                    det.placa(p["eje"], p["pos"], s, 0.3, int(rng.choice([-1, 1])), T, tecla=False)

    # ---- pisos (cada sala partida en rectangulos), techo y luces
    mod.EXTRA_XML.append('<geom name="suelo_exterior" type="plane" pos="0 0 0.002" size="60 60 0.05" '
                         'material="m_suelo" contype="0" conaffinity="0"/>')
    rects = []
    for k, pol in salas.items():
        xs = sorted({round(p[0], 4) for p in pol})
        ys = sorted({round(p[1], 4) for p in pol})
        mat = "piso_m" if rng.random() < 0.5 else "piso_h"
        if realista:
            mat = "piso_h" if tipo_sala[k] in ("Bathroom", "Kitchen") else "piso_m"
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                if punto_en(pol, (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2):
                    mod.solido(mat, f"piso_{k}_{i}_{j}".replace("|", "_"), xs[i], xs[i + 1], ys[j], ys[j + 1],
                               -0.05, mod.SLAB, col=False)
                    rects.append(dict(xa=xs[i], xb=xs[i + 1], ya=ys[j], yb=ys[j + 1]))
    mod.M["techo"].caja(mod.X0 - T / 2, mod.X1 + T / 2, mod.Y0 - T / 2, mod.Y1 + T / 2, H, H + 0.2)
    for r in rects:
        cx, cy = (r["xa"] + r["xb"]) / 2, (r["ya"] + r["yb"]) / 2
        if realista:
            det.lampara_techo(cx, cy, H)
        else:
            mod.M["luz"].caja(cx - 0.25, cx + 0.25, cy - 0.25, cy + 0.25, H - 0.025, H)
    if realista:
        if rng.random() < 0.5:
            dc.textura_ceramica(os.path.join(carpeta, "assets", "piso_hormigon.png"), rng)
        else:
            foto = os.path.join(gv.TEXTURAS, f"piso__{rng.choice(['interior_tiles', 'brown_floor_tiles'])}.jpg")
            gv.abrir_textura(foto).save(os.path.join(carpeta, "assets", "piso_hormigon.png"))
            texturas["pisos"][1] = os.path.basename(foto)

    # ---- muebles
    telas = {}
    for i, (x0, x1, y0, y1, z0, z1, tipo, frente) in enumerate(cajas):
        nombre_m = f"mueble_{i}_{tipo}"
        mod.col_caja(nombre_m, x0, x1, y0, y1, z0, z1)
        if realista:
            det.mueble((x0, x1, y0, y1, z0, z1), tipo, frente)
            continue
        if tipo in MADERA:
            mod.M["madera"].caja(x0, x1, y0, y1, z0, z1)
            continue
        r, g, b = COLORES.get(tipo, (0.6, 0.58, 0.55))
        apariencia = f'rgba="{r} {g} {b} 1"'
        if realista and tipo in TAPIZADOS:
            apariencia = f'material="m_tela_{tipo}"'
            telas[tipo] = (r, g, b)
        mod.EXTRA_XML.append(
            f'<geom name="{nombre_m}" type="box" pos="{(x0 + x1) / 2:.3f} {(y0 + y1) / 2:.3f} {(z0 + z1) / 2:.3f}" '
            f'size="{(x1 - x0) / 2:.3f} {(y1 - y0) / 2:.3f} {(z1 - z0) / 2:.3f}" {apariencia} '
            f'contype="0" conaffinity="0"/>')

    if realista:
        for x0, x1, y0, y1, z0, z1, tipo, frente, clase in extras:
            if clase == "hijo":
                det.objeto((x0, x1, y0, y1, z0, z1), tipo, frente)
            elif tipo == "Television":
                det.objeto((x0, x1, y0, y1, z0, z1), tipo, frente)
            elif tipo == "Mirror":
                det.espejo(x0, x1, y0, y1, z0, z1, "x" if x1 - x0 < y1 - y0 else "y")
            else:
                det.cuadro(x0, x1, y0, y1, z0, z1, "x" if x1 - x0 < y1 - y0 else "y")
    mod.escribir_xml([], [])
    if realista:
        det.inyectar(os.path.join(carpeta, "escenario.xml"))
    if telas:
        ruta = os.path.join(carpeta, "escenario.xml")
        with open(ruta, encoding="utf-8") as f:
            xml = f.read()
        extra = [f'<texture name="t_tela" type="2d" file="{mod.PREF}/assets/tela.png"/>']
        extra += [f'<material name="m_tela_{t}" texture="t_tela" texuniform="true" texrepeat="6 6" '
                  f'rgba="{r} {g} {b} 1" specular="0.05" shininess="0.05"/>' for t, (r, g, b) in telas.items()]
        xml = xml.replace("  </asset>", "".join("    " + e + "\n" for e in extra) + "  </asset>", 1)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(xml)
    gv.ajustar_luces(os.path.join(carpeta, "escenario.xml"), rng, dict(salas=rects), True)
    if realista:
        # luz de relleno neutra y mas alta: en una casa real la luz rebota en paredes y techo
        ruta = os.path.join(carpeta, "escenario.xml")
        with open(ruta, encoding="utf-8") as f:
            xml = f.read()
        amb, dif = rng.uniform(0.36, 0.48), rng.uniform(0.15, 0.25)
        xml = re.sub(r'<headlight ambient="[^"]*" diffuse="[^"]*"',
                     f'<headlight ambient="{amb:.2f} {amb:.2f} {amb * 0.97:.2f}" diffuse="{dif:.2f} {dif:.2f} {dif:.2f}"', xml)
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(xml)

    with open(os.path.join(carpeta, "interior.json"), "w", encoding="utf-8") as f:
        json.dump([[list(p) for p in pol] for pol in salas.values()], f)
    danos = [d for d in mod.MANIFEST if "pieza" in d]
    with open(os.path.join(carpeta, "variante.json"), "w", encoding="utf-8") as f:
        json.dump(dict(nombre=nombre, origen="ProcTHOR procthor-10k, split train", casa=indice, estilo=estilo,
                       salas=[tipo_sala[k] for k in salas], texturas=texturas, holgura_origen=round(holgura, 2),
                       limites=[mod.X0, mod.X1, mod.Y0, mod.Y1], danos_de_pared=len(danos), muebles=len(cajas)),
                  f, indent=2, ensure_ascii=False)
    print(f"{nombre} (casa {indice}): {len(salas)} salas {sorted(tipo_sala.values())}, "
          f"{mod.X1 - mod.X0:.1f} x {mod.Y1 - mod.Y0:.1f} m, {len(paredes)} paredes, {len(danos)} daños, "
          f"{len(cajas)} muebles, holgura en el origen {holgura:.2f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--casas", type=int, nargs="+", required=True, help="indices dentro de casas_60.json")
    ap.add_argument("--estilo", choices=["original", "realista"], default="original")
    args = ap.parse_args()
    os.makedirs(gv.DESTINO, exist_ok=True)
    for n, i in enumerate(args.casas, start=1):
        generar(i, n, args.estilo)


if __name__ == "__main__":
    main()
