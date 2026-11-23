"""
Genera edificios distintos a partir del generador de edificio_informe, para
tener variedad de fotos simuladas al entrenar el detector de daños.

No modifica escenarios/edificio_informe: carga generar_edificio.py como
libreria, le cambia los parametros y escribe cada edificio en su carpeta:

    escenarios/variantes/edificio_vNN/
        escenario.xml, meshes/, assets/, damage_manifest.json, variante.json

Cada edificio cambia: medidas, cantidad y forma de las salas, puertas y
ventanas, texturas de paredes, pisos y muebles, color de pintura, muebles,
luz, techo, y la cantidad, el tipo, la forma y el lugar de los daños.

Los edificios se pueden abrir con launch_g1_nav.py poniendo
SCENARIO = "variantes/edificio_v01". El origen (0, 0) siempre queda libre.

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/generar_variantes.py --desde 1 --hasta 24
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import re
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGINAL = os.path.join(RAIZ, "escenarios", "edificio_informe")
TEXTURAS = os.path.join(RAIZ, "dataset_sim", "texturas")
DESTINO = os.path.join(RAIZ, "escenarios", "variantes")
LADO_TEX = 1024
PINTURAS = [(232, 232, 228), (196, 196, 194), (236, 226, 204), (214, 220, 226), (222, 230, 214),
            (238, 214, 196), (204, 216, 232), (176, 176, 172), (240, 236, 220), (226, 204, 180)]
PROB_PINTAR = 0.65     # proporcion de edificios con paredes pintadas de un color plano
OVERLAY_REAL = None    # si se define, funcion (tipo, ancho, alto, rng) -> overlay hecho con fotos reales
PAREDES_CLARAS = ("white_plaster_02", "beige_wall_001", "grey_plaster", "grey_plaster_02", "plastered_wall",
                  "plastered_wall_02", "white_stucco", "white_stucco_02")


def cargar_generador(nombre):
    spec = importlib.util.spec_from_file_location(f"gen_{nombre}", os.path.join(ORIGINAL, "generar_edificio.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def abrir_textura(ruta):
    return Image.open(ruta).convert("RGB").resize((LADO_TEX, LADO_TEX), Image.LANCZOS)


def pintar(img, color, fuerza):
    """Pared pintada: deja solo el grano fino de la textura sobre un color plano."""
    a = np.asarray(img).astype(np.float32)
    g = a.mean(axis=2)
    bajo = gaussian_filter(g, sigma=20, mode="wrap")
    hp = g / np.maximum(bajo, 1.0) - 1.0
    out = np.array(color, np.float32)[None, None, :] * (1.0 + fuerza * hp)[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def preparar_texturas(rng, assets):
    """Elige y escribe las texturas del edificio. Devuelve lo elegido."""
    def elegir(tipo, solo=None):
        rutas = sorted(glob.glob(os.path.join(TEXTURAS, f"{tipo}__*.jpg")))
        if solo:
            rutas = [r for r in rutas if any(s in r for s in solo)]
        return rutas[rng.integers(len(rutas))]

    elegido = {}
    if rng.random() < PROB_PINTAR:
        ruta = elegir("pared", PAREDES_CLARAS)
        color = np.array(PINTURAS[rng.integers(len(PINTURAS))]) * rng.uniform(0.9, 1.04)
        pintar(abrir_textura(ruta), color, rng.uniform(0.2, 0.7)).save(os.path.join(assets, "pared_revoque.png"))
        elegido["pared"] = f"{os.path.basename(ruta)} pintada {np.round(color).astype(int).tolist()}"
    else:
        ruta = elegir("pared")
        abrir_textura(ruta).save(os.path.join(assets, "pared_revoque.png"))
        elegido["pared"] = os.path.basename(ruta)
    ruta = elegir("pared", ("concrete", "brick", "grey_plaster"))
    abrir_textura(ruta).save(os.path.join(assets, "pared_hormigon.png"))
    elegido["pared_secundaria"] = os.path.basename(ruta)
    pisos = sorted(glob.glob(os.path.join(TEXTURAS, "piso__*.jpg")))
    a, b = rng.choice(len(pisos), 2, replace=False)
    abrir_textura(pisos[a]).save(os.path.join(assets, "piso_hormigon.png"))
    abrir_textura(pisos[b]).save(os.path.join(assets, "piso_madera.png"))
    elegido["pisos"] = [os.path.basename(pisos[a]), os.path.basename(pisos[b])]
    ruta = elegir("mueble")
    abrir_textura(ruta).save(os.path.join(assets, "mueble_madera.png"))
    elegido["mueble"] = os.path.basename(ruta)
    for nombre in ("ladrillo.png", "escombro.png"):
        shutil.copy(os.path.join(ORIGINAL, "assets", nombre), os.path.join(assets, nombre))
    return elegido


def lejos(valor, otros, minimo):
    return all(abs(valor - o) >= minimo for o in otros)


def armar_planta(rng):
    """Planta rectangular dividida en columnas; cada columna puede partirse en dos salas."""
    ancho, fondo = rng.uniform(10, 18), rng.uniform(8, 13)
    x0 = -ancho / 2 + rng.uniform(-1.5, 1.5)
    y0 = -rng.uniform(2.5, 4.0)
    x1, y1 = x0 + ancho, y0 + fondo
    cortes_x = []
    for _ in range(40):
        if len(cortes_x) >= int(ancho // 4.5):
            break
        c = round(rng.uniform(x0 + 3, x1 - 3), 1)
        if abs(c) >= 1.2 and lejos(c, cortes_x, 3.2):
            cortes_x.append(c)
    cortes_x.sort()
    bordes = [x0] + cortes_x + [x1]
    candidatos_y = [y for y in np.arange(y0 + 3, y1 - 2.9, 3.0) if abs(y) >= 1.2]
    columnas = []
    for i in range(len(bordes) - 1):
        corte = None
        if candidatos_y and rng.random() < 0.65:
            corte = float(candidatos_y[rng.integers(len(candidatos_y))])
        columnas.append(dict(xa=bordes[i], xb=bordes[i + 1], corte=corte))
    salas = []
    for col in columnas:
        tramos = [(y0, y1)] if col["corte"] is None else [(y0, col["corte"]), (col["corte"], y1)]
        for ya, yb in tramos:
            salas.append(dict(xa=col["xa"], xb=col["xb"], ya=ya, yb=yb))
    return dict(x0=x0, x1=x1, y0=y0, y1=y1, cortes_x=cortes_x, columnas=columnas, salas=salas)


def tramos_libres(a0, a1, aberturas, margen=0.25):
    """Tramos de pared de altura completa, entre aberturas."""
    libres, cur = [], a0 + margen
    for c, w, *_ in sorted(aberturas):
        if c - w / 2 - margen > cur:
            libres.append((cur, c - w / 2 - margen))
        cur = c + w / 2 + margen
    if a1 - margen > cur:
        libres.append((cur, a1 - margen))
    return libres


def construir(mod, rng, planta, techo):
    T, H = mod.T, mod.H
    x0, x1, y0, y1 = planta["x0"], planta["x1"], planta["y0"], planta["y1"]
    ex0, ex1 = x0 - T / 2, x1 + T / 2
    puertas = []   # (x, y, eje de la pared)

    # ---- paredes y aberturas
    cortes_y_todos = sorted({c["corte"] for c in planta["columnas"] if c["corte"] is not None})

    def ventanas(a0, a1, perpendiculares):
        lista = []
        for c in np.arange(a0 + 2.0, a1 - 1.9, 3.4):
            c = round(float(c + rng.uniform(-0.5, 0.5)), 2)
            if rng.random() < 0.6 and lejos(c, perpendiculares, 1.3):
                lista.append(mod.ventana(c))
        return lista

    paredes = [
        dict(nombre="ext_sur", eje="x", pos=y0, a0=ex0, a1=ex1, cara="Y", ab=ventanas(x0, x1, planta["cortes_x"])),
        dict(nombre="ext_norte", eje="x", pos=y1, a0=ex0, a1=ex1, cara="y", ab=ventanas(x0, x1, planta["cortes_x"])),
        dict(nombre="ext_oeste", eje="y", pos=x0, a0=y0 + T / 2, a1=y1 - T / 2, cara="X",
             ab=ventanas(y0, y1, [planta["columnas"][0]["corte"] or 99])),
        dict(nombre="ext_este", eje="y", pos=x1, a0=y0 + T / 2, a1=y1 - T / 2, cara="x",
             ab=ventanas(y0, y1, [planta["columnas"][-1]["corte"] or 99])),
    ]
    for i, c in enumerate(planta["cortes_x"]):
        izq, der = planta["columnas"][i]["corte"], planta["columnas"][i + 1]["corte"]
        quiebres = sorted({y0, y1} | {q for q in (izq, der) if q is not None})
        ab = []
        for a, b in zip(quiebres[:-1], quiebres[1:]):
            if b - a >= 2.4:
                yd = round(float(rng.uniform(a + 1.0, b - 1.0)), 2)
                ab.append(mod.puerta(yd))
                puertas.append((c, yd, "y"))
        paredes.append(dict(nombre=f"int_x_{i}", eje="y", pos=c, a0=y0, a1=y1, cara=None, ab=ab))
    for i, col in enumerate(planta["columnas"]):
        if col["corte"] is None:
            continue
        xd = round(float(rng.uniform(col["xa"] + 1.0, col["xb"] - 1.0)), 2)
        puertas.append((xd, col["corte"], "x"))
        paredes.append(dict(nombre=f"int_y_{i}", eje="x", pos=col["corte"], a0=col["xa"], a1=col["xb"], cara=None,
                            ab=[mod.puerta(xd)]))

    # ---- daños de pared
    def overlay(tipo):
        semilla = int(rng.integers(1, 100000))
        brillo = rng.uniform(0.7, 1.5)
        tinte = rng.uniform(0.85, 1.15, 3)
        if tipo == "grieta_estructural":
            ramas = int(rng.integers(1, 5))
            base = lambda W, Hh: mod.ov_grieta(W, Hh, semilla, ramas)
        elif tipo == "humedad":
            base = lambda W, Hh: mod.ov_mancha(W, Hh, semilla)
        else:
            base = lambda W, Hh: mod.ov_desprendimiento(W, Hh, semilla)

        def ov(W, Hh):
            rgb, a = base(max(8, W), max(8, Hh))
            return np.clip(rgb * brillo * tinte, 0, 255), a
        return ov

    for p in paredes:
        p["danos"] = []
    cantidad = int(rng.integers(6, 11))
    for _ in range(cantidad * 6):
        if sum(len(p["danos"]) for p in paredes) >= cantidad:
            break
        p = paredes[rng.integers(len(paredes))]
        tipo = rng.choice(["grieta_estructural", "humedad", "desprendimiento_revoque"], p=[0.5, 0.25, 0.25])
        if tipo == "grieta_estructural":
            w, h = rng.uniform(0.8, 1.8), rng.uniform(1.4, 2.6)
        elif tipo == "humedad":
            w, h = rng.uniform(1.0, 2.2), rng.uniform(1.0, 2.0)
        else:
            w, h = rng.uniform(0.6, 1.5), rng.uniform(0.6, 1.4)
        # los cruces con otras paredes cuentan como aberturas: ahi no se ve la cara
        cruces = planta["cortes_x"] if p["eje"] == "x" else cortes_y_todos
        ocupado = list(p["ab"]) + [(c, 0.3) for c in cruces if p["a0"] < c < p["a1"]]
        libres = [(a, b) for a, b in tramos_libres(p["a0"], p["a1"], ocupado) if b - a >= w + 0.2]
        if not libres:
            continue
        a, b = libres[rng.integers(len(libres))]
        s = float(rng.uniform(a + w / 2, b - w / 2))
        if not lejos(s, [d["s"] for d in p["danos"]], 2.4):
            continue
        z = float(rng.uniform(h / 2 + 0.05, H - h / 2 - 0.02)) if tipo != "grieta_estructural" else H - h / 2 - 0.01
        cara = p["cara"] or (rng.choice(["x", "X"]) if p["eje"] == "y" else rng.choice(["y", "Y"]))
        p["danos"].append(dict(cara=str(cara), s=s, z=z, w=float(w), h=float(h), ov=OVERLAY_REAL(str(tipo), float(w), float(h), rng) if OVERLAY_REAL else overlay(tipo), tipo=str(tipo),
                               sala=p["nombre"]))

    for p in paredes:
        mat = "revoque"
        if not p["danos"] and p["cara"] is None and rng.random() < 0.25:
            mat = "hormigon"
        mod.pared(p["nombre"], p["eje"], p["pos"], p["a0"], p["a1"], p["ab"], mat=mat, danos=p["danos"])

    # ---- zocalos, pisos y techo
    if rng.random() < 0.7:
        e, z = 0.012, 0.10
        for eje, pos, a0, a1 in (("x", y0, x0, x1), ("x", y1, x0, x1), ("y", x0, y0, y1), ("y", x1, y0, y1)):
            for lado in (-1, 1):
                off = pos + lado * (T / 2 + e / 2)
                if eje == "x":
                    mod.M["marco"].caja(a0 + 0.1, a1 - 0.1, off - e / 2, off + e / 2, mod.SLAB, z)
                else:
                    mod.M["marco"].caja(off - e / 2, off + e / 2, a0 + 0.1, a1 - 0.1, mod.SLAB, z)
    mod.EXTRA_XML.append('<geom name="suelo_exterior" type="plane" pos="0 0 0.002" size="40 40 0.05" '
                         'material="m_suelo" contype="0" conaffinity="0"/>')
    for i, s in enumerate(planta["salas"]):
        mod.solido("piso_m" if rng.random() < 0.5 else "piso_h", f"piso_{i}", s["xa"], s["xb"], s["ya"], s["yb"],
                   -0.05, mod.SLAB, col=False)
    if techo:
        mod.M["techo"].caja(ex0, ex1, y0 - T / 2, y1 + T / 2, H, H + 0.2)
        for s in planta["salas"]:
            cx, cy = (s["xa"] + s["xb"]) / 2, (s["ya"] + s["yb"]) / 2
            mod.M["luz"].caja(cx - 0.3, cx + 0.3, cy - 0.3, cy + 0.3, H - 0.025, H)

    # ---- hojas de puerta
    for i, (px, py, eje) in enumerate(puertas):
        if rng.random() < 0.7:
            abre = rng.uniform(40, 85) * rng.choice([-1, 1])
            if eje == "y":
                mod.hoja_puerta(f"puerta_{i}", px, py - 0.5, 90 - abre)
            else:
                mod.hoja_puerta(f"puerta_{i}", px - 0.5, py, abre)

    # ---- muebles
    ocupados = [(0.0, 0.0, 1.3)] + [(px, py, 1.4) for px, py, _ in puertas]

    def libre(cx, cy, radio):
        return all(math.hypot(cx - ox, cy - oy) >= radio + r for ox, oy, r in ocupados)

    k = 0
    for s in planta["salas"]:
        area = (s["xb"] - s["xa"]) * (s["yb"] - s["ya"])
        for _ in range(int(rng.integers(0, max(2, int(area / 7)) + 1))):
            tipo = rng.choice(["mesa", "escritorio", "estante", "caja"], p=[0.3, 0.3, 0.2, 0.2])
            for _intento in range(12):
                if tipo == "estante":
                    lado = rng.choice([-1, 1])
                    cx = round(float(s["xa"] + T / 2 + 0.22 if lado < 0 else s["xb"] - T / 2 - 0.22), 2)
                    cy = round(float(rng.uniform(s["ya"] + 1.0, s["yb"] - 1.0)), 2)
                    radio = 0.7
                else:
                    cx = round(float(rng.uniform(s["xa"] + 1.3, s["xb"] - 1.3)), 2)
                    cy = round(float(rng.uniform(s["ya"] + 1.3, s["yb"] - 1.3)), 2)
                    radio = 1.3 if tipo != "caja" else 0.5
                if not (s["xa"] < cx < s["xb"] and s["ya"] < cy < s["yb"]) or not libre(cx, cy, radio):
                    continue
                if tipo == "mesa":
                    lx, ly = round(float(rng.uniform(1.0, 1.9)), 2), round(float(rng.uniform(0.7, 1.0)), 2)
                    mod.mesa(cx, cy, lx, ly)
                    for dx in (-0.45, 0.45)[: int(rng.integers(0, 3))]:
                        mod.silla(round(cx + dx, 2), round(cy + ly / 2 + 0.35, 2), dir_y=-1)
                elif tipo == "escritorio":
                    mod.escritorio(cx, cy)
                elif tipo == "estante":
                    mod.estante(cx, cy, ly=round(float(rng.uniform(0.9, 1.6)), 2), h=round(float(rng.uniform(1.4, 2.1)), 2))
                else:
                    mod.caja_carton(f"carton_{k}", cx, cy, yaw=float(rng.uniform(0, 90)))
                ocupados.append((cx, cy, radio))
                k += 1
                break

    # ---- escombros (pocos: las fotos donde se ven no se usan para entrenar)
    rocas, poses = [], []
    if rng.random() < 0.35:
        s = planta["salas"][rng.integers(len(planta["salas"]))]
        for _intento in range(12):
            cx, cy = rng.uniform(s["xa"] + 1.2, s["xb"] - 1.2), rng.uniform(s["ya"] + 1.2, s["yb"] - 1.2)
            if libre(cx, cy, 1.0):
                rocas = mod.crear_rocas(8)
                poses = mod.asentar_escombros(rocas, [(float(cx), float(cy), 0.4, 1.6, 0.6, int(rng.integers(6, 16)))])
                mod.MANIFEST.append(dict(tipo="escombros", sala="-", centro=[float(cx), float(cy)]))
                break
    return rocas, poses


def ajustar_luces(ruta_xml, rng, planta, techo):
    with open(ruta_xml, encoding="utf-8") as f:
        xml = f.read()
    # MuJoCo busca las mallas desde unitree_g1/assets y las texturas desde la carpeta de este xml
    xml = re.sub(r'(<texture [^>]*file=")\.\./\.\./escenarios/variantes/', r'\g<1>../../variantes/', xml)
    xml = re.sub(r'\s*<light name="luz_\d+"[^>]*/>', "", xml)
    xml = re.sub(r'\s*<light name="(sol|relleno)"[^>]*/>', "", xml)
    calido = rng.uniform(0.85, 1.0)
    luces = []
    for i, s in enumerate(planta["salas"]):
        d = rng.uniform(0.35, 1.0) if rng.random() < 0.85 else 0.0
        luces.append(f'<light name="luz_{i}" pos="{(s["xa"] + s["xb"]) / 2:.2f} {(s["ya"] + s["yb"]) / 2:.2f} 2.65" '
                     f'dir="0 0 -1" cutoff="110" exponent="1" diffuse="{d:.2f} {d:.2f} {d * calido:.2f}" '
                     f'specular="0.1 0.1 0.1" attenuation="1 0.03 0.01" castshadow="false"/>')
    ang = rng.uniform(0, 2 * math.pi)
    sol = rng.uniform(0.7, 1.6)
    luces.append(f'<light name="sol" directional="true" pos="0 0 20" dir="{0.65 * math.cos(ang):.2f} '
                 f'{0.65 * math.sin(ang):.2f} -{rng.uniform(0.4, 0.8):.2f}" diffuse="{sol:.2f} {sol * 0.9:.2f} '
                 f'{sol * rng.uniform(0.65, 0.9):.2f}" specular="0.2 0.2 0.2" castshadow="true"/>')
    luces.append('<light name="relleno" directional="true" pos="0 0 10" dir="-0.2 0.1 -1" '
                 'diffuse="0.12 0.13 0.16" castshadow="false"/>')
    amb = rng.uniform(0.18, 0.42) if not techo else rng.uniform(0.25, 0.5)
    xml = re.sub(r'<headlight ambient="[^"]*" diffuse="[^"]*"',
                 f'<headlight ambient="{amb:.2f} {amb:.2f} {amb + 0.02:.2f}" diffuse="{rng.uniform(0.08, 0.3):.2f} '
                 f'{rng.uniform(0.08, 0.3):.2f} 0.15"', xml)
    xml = xml.replace("<worldbody>", "<worldbody>\n    " + "\n    ".join(luces), 1)
    with open(ruta_xml, "w", encoding="utf-8") as f:
        f.write(xml)


def generar(numero, nombre=None, techo=None):
    nombre = nombre or f"edificio_v{numero:02d}"
    carpeta = os.path.join(DESTINO, nombre)
    if os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
    os.makedirs(os.path.join(carpeta, "assets"))
    os.makedirs(os.path.join(carpeta, "meshes"))
    rng = np.random.default_rng(1000 + numero)

    texturas = preparar_texturas(rng, os.path.join(carpeta, "assets"))
    planta = armar_planta(rng)
    azar = bool(rng.random() < 0.5)
    techo = azar if techo is None else techo

    mod = cargar_generador(nombre)
    mod.AQUI, mod.ASSETS, mod.MESHES = carpeta, os.path.join(carpeta, "assets"), os.path.join(carpeta, "meshes")
    mod.PREF = f"../../escenarios/variantes/{nombre}"
    mod.RNG = np.random.default_rng(2000 + numero)
    mod.X0, mod.X1, mod.Y0, mod.Y1 = planta["x0"], planta["x1"], planta["y0"], planta["y1"]
    mod.TECHO, mod.CAMARAS, mod.PX_PIEZA = techo, [], 256

    rocas, poses = construir(mod, rng, planta, techo)
    mod.escribir_xml(rocas, poses)
    ajustar_luces(os.path.join(carpeta, "escenario.xml"), rng, planta, techo)

    danos = [d for d in mod.MANIFEST if "pieza" in d]
    with open(os.path.join(carpeta, "variante.json"), "w", encoding="utf-8") as f:
        json.dump(dict(nombre=nombre, semilla=1000 + numero, techo=techo, texturas=texturas,
                       limites=[planta["x0"], planta["x1"], planta["y0"], planta["y1"]], salas=planta["salas"],
                       danos_de_pared=len(danos)), f, indent=2, ensure_ascii=False)
    print(f"{nombre}: {planta['x1'] - planta['x0']:.1f} x {planta['y1'] - planta['y0']:.1f} m, "
          f"{len(planta['salas'])} salas, {len(danos)} daños de pared, techo {'si' if techo else 'no'}, "
          f"pared {texturas['pared']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1)
    ap.add_argument("--hasta", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(DESTINO, exist_ok=True)
    for n in range(args.desde, args.hasta + 1):
        generar(n)


if __name__ == "__main__":
    main()
