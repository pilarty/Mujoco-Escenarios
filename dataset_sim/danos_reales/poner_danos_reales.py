"""
Arma casa_blanca_fotos: la casa blanca con daños sacados de fotos reales en
vez de los dibujados por danos_realistas.py, y con daños tambien en el piso.
Es solo para probar el modelo en el simulador, no para entrenar.

Como se pega un daño real sin que se note el recuadro:
  1. de la foto se toma el recuadro del daño (elegidas.csv) agrandado un 30 %,
     para que entre la pared sana de alrededor, y se ajusta al tamaño del daño
     en la pared, sin deformarlo; se descartan las fotos donde el daño llega
     al borde del recorte (se veria el recuadro);
  2. se estima el color de la pared sana de la foto (la envolvente clara,
     suavizada) y se divide la foto por ese color: queda solo cuanto oscurece
     o cambia de color el daño;
  3. ese cambio se aplica sobre la pintura (o el piso) de la casa, con los
     bordes que se desvanecen de a poco. El grano fino de la foto se ignora.
Asi la grieta, la mancha o el revoque caido son de una foto real, pero la
pintura y la luz son las de la casa.

Paredes: se usan los mismos 8 lugares, tamaños y tipos de daño de la casa
blanca. Pisos: se agregan 6 daños (grietas y manchas de agua) en las salas
que recorre el robot, lejos de los muebles.

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/danos_reales/poner_danos_reales.py
Salida: escenarios/variantes/casa_blanca_fotos/ (con danos_piso.json)
"""
import csv
import functools
import json
import math
import os
import shutil
import sys

import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, zoom

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import generar_capturas as gc  # noqa: E402
import recorrido_robot as rr  # noqa: E402

ORIGEN, NOMBRE = "casa_blanca", "casa_blanca_fotos"
SELECCION = {
    "grieta_estructural": [0, 1, 10, 13, 14, 16, 18, 22, 31, 37, 42, 47, 48, 49, 56, 59, 60, 61, 62, 64, 65, 68,
                           74, 76, 77, 80, 82, 84, 87, 88, 92, 93, 94, 95, 99, 100, 103, 107, 109, 115, 118, 119],
    "humedad": [2, 3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 17, 19, 21, 23, 25, 27, 28, 32, 35, 36, 37, 38, 39, 41, 45,
                46, 47, 49, 53, 54, 57, 59, 60, 63, 64, 67, 68, 71, 73, 75, 76, 77, 79, 80, 82, 84, 86, 89, 91, 96,
                98, 100, 101, 103, 107, 108, 109, 110, 111, 113, 115],
    "desprendimiento_revoque": [4, 7, 16, 18, 19, 22, 23, 27, 30, 31, 36, 38, 43, 44, 46, 47, 51, 52, 62, 63, 70,
                                76, 80, 87, 92, 93, 94, 100, 108, 110, 114, 116],
}
MANCHAS_DE_AGUA = [3, 17, 25, 27, 91, 100, 101, 110]   # de la lista de humedad, para el piso
PX_POR_M_PISO = 512
TILE = 2.0


def recortes():
    with open(os.path.join(AQUI, "elegidas.csv"), encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    por_tipo = {}
    for tipo, indices in SELECCION.items():
        del_tipo = [r for r in filas if r["tipo"] == tipo]
        por_tipo[tipo] = [del_tipo[i] for i in indices if i < len(del_tipo)]
    return por_tipo


@functools.lru_cache(maxsize=None)
def _foto(archivo):
    foto = Image.open(os.path.join(AQUI, "fotos", archivo)).convert("RGB")
    foto.thumbnail((900, 900))
    return foto


def abrir_recorte(r, margen=0.3):
    """El recuadro del daño agrandado un 30 %, para que entre la pared sana de alrededor y el borde
    del daño sea el de la foto."""
    foto = _foto(r["archivo"])
    W, H = foto.size
    x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
    w, h = w * (1 + margen), h * (1 + margen)
    caja = (max(0, int((x - w / 2) * W)), max(0, int((y - h / 2) * H)), min(W, int((x + w / 2) * W)),
            min(H, int((y + h / 2) * H)))
    return foto.crop(caja)


UMBRAL = {"grieta_estructural": (0.15, 0.35), "humedad": (0.06, 0.20), "desprendimiento_revoque": (0.08, 0.22)}


def cambio_del_dano(recorte, ancho_px, alto_px, tipo):
    """Factor por pixel (alto, ancho, 3) que multiplica a la pintura: 1 donde no hay daño.
    Los cambios chicos (el grano de la superficie de la foto) no pasan; el umbral depende del tipo."""
    lado = max(ancho_px / recorte.width, alto_px / recorte.height)
    img = recorte.resize((max(8, int(recorte.width * lado)), max(8, int(recorte.height * lado))), Image.LANCZOS)
    x0, y0 = (img.width - ancho_px) // 2, (img.height - alto_px) // 2
    c = np.asarray(img.crop((x0, y0, x0 + ancho_px, y0 + alto_px)), np.float32) + 1.0
    # color de la pared sana: envolvente clara suavizada, calculada a un cuarto del tamaño
    chica = c[::4, ::4]
    t = max(2, min(chica.shape[:2]) // 6)
    env = np.stack([gaussian_filter(grey_dilation(chica[..., k], size=(t, t)), t) for k in range(3)], axis=-1)
    sano = zoom(env, (alto_px / env.shape[0], ancho_px / env.shape[1], 1), order=1)[:alto_px, :ancho_px]
    factor = gaussian_filter(np.clip(c / np.maximum(sano, 1.0), 0.05, 1.25), (0.8, 0.8, 0))
    desvio = np.abs(factor - 1).max(axis=-1)
    u0, u1 = UMBRAL[tipo]
    peso = gaussian_filter(np.clip((desvio - u0) / (u1 - u0), 0, 1), 1.5)[..., None]
    yy, xx = np.mgrid[0:alto_px, 0:ancho_px].astype(np.float32)
    borde = np.minimum.reduce([xx, ancho_px - 1 - xx, yy, alto_px - 1 - yy]) / (0.15 * min(ancho_px, alto_px))
    borde = (0.5 - 0.5 * np.cos(np.pi * np.clip(borde, 0, 1)))[..., None]
    anillo = borde[..., 0] < 0.5
    return 1 + (factor - 1) * peso * borde, float(peso[..., 0][anillo].mean()), float(peso[..., 0][~anillo].mean())


USADAS = set()   # cada foto se usa una sola vez


def elegir(opciones, ancho_px, alto_px, tipo, rng):
    """Prueba fotos al azar hasta encontrar una cuyo daño no toque el borde del recorte
    (si lo toca, se veria el recuadro) y que tenga un daño visible en el medio."""
    mejor = None
    libres = [o for o in opciones if o["archivo"] not in USADAS] or opciones
    for _ in range(60):
        r = libres[rng.integers(len(libres))]
        factor, en_borde, adentro = cambio_del_dano(abrir_recorte(r), ancho_px, alto_px, tipo)
        puntaje = adentro - 3 * en_borde
        if mejor is None or puntaje > mejor[2]:
            mejor = (r, factor, puntaje)
        if en_borde < 0.12 and adentro > 0.15:
            break
    USADAS.add(mejor[0]["archivo"])
    return mejor[0], mejor[1]


def paredes(carpeta, modelo, datos, nombres, rng, por_tipo):
    with open(os.path.join(carpeta, "damage_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    base = np.array(Image.open(os.path.join(carpeta, "assets", "pared_revoque.png")).convert("RGB"), np.float32)
    S = base.shape[0]
    usados = []
    for pieza in sorted({d["pieza"] for d in manifest if "pieza" in d}):
        ruta = os.path.join(carpeta, "assets", f"pieza_{pieza}.png")
        atlas = np.array(Image.open(ruta).convert("RGB"), np.float32)
        Hp, Wp = atlas.shape[0] // 2, atlas.shape[1]
        gid = [i for i, n in enumerate(nombres) if n.startswith(f"vis_pieza_{pieza}")][0]
        centro = datos.geom_xpos[gid] + datos.geom_xmat[gid].reshape(3, 3) @ modelo.geom_aabb[gid][:3]
        medio = np.abs(datos.geom_xmat[gid].reshape(3, 3) @ modelo.geom_aabb[gid][3:])
        largo = 1 - int(np.argmin(medio[:2]))
        s0, s1 = centro[largo] - medio[largo], centro[largo] + medio[largo]
        z0, z1 = centro[2] - medio[2], centro[2] + medio[2]
        sm = s0 + (np.arange(Wp) + 0.5) / Wp * (s1 - s0)
        zm = z1 - (np.arange(Hp) + 0.5) / Hp * (z1 - z0)
        col = ((sm / TILE) % 1.0 * S).astype(int) % S
        fil = ((1.0 - (zm / TILE) % 1.0) * S).astype(int) % S
        sana = base[fil[:, None], col[None, :]]
        nuevo = np.concatenate([sana, sana], axis=0)
        for d in manifest:
            if d.get("pieza") != pieza:
                continue
            mitad = 0 if d["cara"].isupper() else Hp
            c0 = int(np.searchsorted(sm, d["centro_s"] - d["ancho"] / 2))
            c1 = int(np.searchsorted(sm, d["centro_s"] + d["ancho"] / 2))
            f0 = int(np.searchsorted(-zm, -(d["centro_z"] + d["alto"] / 2)))
            f1 = int(np.searchsorted(-zm, -(d["centro_z"] - d["alto"] / 2)))
            if c1 - c0 < 8 or f1 - f0 < 8:
                continue
            r, factor = elegir(por_tipo[d["tipo"]], c1 - c0, f1 - f0, d["tipo"], rng)
            nuevo[mitad + f0:mitad + f1, c0:c1] *= factor
            usados.append(dict(lugar=f"pared {pieza}", tipo=d["tipo"], foto=r["archivo"]))
        Image.fromarray(np.clip(nuevo, 0, 255).astype(np.uint8)).save(ruta)
    return usados


def pisos(carpeta, modelo, datos, nombres, rng, por_tipo):
    """Agrega daños en el piso como una lamina fina con la textura del piso y el daño encima."""
    with open(os.path.join(carpeta, "interior.json"), encoding="utf-8") as f:
        interior = json.load(f)
    with open(os.path.join(carpeta, "variante.json"), encoding="utf-8") as f:
        salas = json.load(f)["salas"]
    p = rr.plano(modelo, datos, nombres, interior)
    alcanzables = rr.libre_cerca(p, 0.0, 0.0, rr.RADIO_ROBOT)
    from scipy.ndimage import label
    comp, _ = label(p["dist"] >= rr.RADIO_ROBOT)
    principal = comp[alcanzables]
    texturas = {m: np.array(Image.open(os.path.join(carpeta, "assets", f)).convert("RGB"), np.float32)
                for m, f in (("h", "piso_hormigon.png"), ("m", "piso_madera.png"))}
    manchas = [r for i, r in zip(SELECCION["humedad"], por_tipo["humedad"]) if i in MANCHAS_DE_AGUA]
    danos, xml_assets, xml_geoms, usados = [], [], [], []
    puestos = []
    for n in range(40):
        if len(danos) >= 6:
            break
        tipo = "grieta_estructural" if n % 2 == 0 else "humedad"
        w, h = rng.uniform(0.7, 1.3), rng.uniform(0.5, 1.0)
        candidatas = np.argwhere((comp == principal) & (p["dist"] >= max(w, h) / 2 + 0.05))
        if not len(candidatas):
            continue
        i, j = candidatas[rng.integers(len(candidatas))]
        cx, cy = rr.punto(p, i, j)
        if any(math.hypot(cx - a, cy - b) < 2.5 for a, b in puestos):
            continue
        k = p["sala"][i, j]
        gid = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(modelo, datos, np.array([cx, cy, 0.5]), np.array([0.0, 0.0, -1.0]),
                      np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, gid)
        debajo = nombres[gid[0]] if gid[0] >= 0 else ""
        if debajo not in ("vis_piso_h", "vis_piso_m"):
            continue
        mat = debajo[-1]
        x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
        Wd, Hd = int(w * PX_POR_M_PISO), int(h * PX_POR_M_PISO)
        xs = x0 + (np.arange(Wd) + 0.5) / PX_POR_M_PISO
        ys = y1 - (np.arange(Hd) + 0.5) / PX_POR_M_PISO
        T = texturas[mat]
        S = T.shape[0]
        col = ((xs / TILE) % 1.0 * S).astype(int) % S
        fil = ((1.0 - (ys / TILE) % 1.0) * S).astype(int) % S
        piso = T[fil[:, None], col[None, :]]
        # sobre la madera o la ceramica muchos daños casi no se notan: se busca uno que cambie al menos el 3 %
        for _ in range(6):
            r, factor = elegir(por_tipo["grieta_estructural"] if tipo == "grieta_estructural" else manchas, Wd, Hd,
                               tipo, rng)
            img = np.clip(piso * factor, 0, 255).astype(np.uint8)
            if (np.abs(img.astype(np.float32) - piso).max(axis=-1) > 14).mean() >= 0.03:
                break
        else:
            continue
        nombre = f"dano_piso_{len(danos)}"
        Image.fromarray(img).save(os.path.join(carpeta, "assets", f"{nombre}.png"))
        mascara = np.abs(img.astype(np.float32) - piso).max(axis=-1) > 14
        Image.fromarray((mascara * 255).astype(np.uint8)).save(os.path.join(carpeta, "assets", f"{nombre}_mascara.png"))
        # lamina de 1 mm (MuJoCo no acepta mallas planas); la cara de arriba lleva la textura
        with open(os.path.join(carpeta, "meshes", f"{nombre}.obj"), "w") as f:
            for z in (0.0048, 0.0038):
                f.write(f"v {x0:.4f} {y0:.4f} {z}\nv {x1:.4f} {y0:.4f} {z}\nv {x1:.4f} {y1:.4f} {z}\n"
                        f"v {x0:.4f} {y1:.4f} {z}\n")
            f.write("vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n")
            f.write("f 1/1 2/2 3/3\nf 1/1 3/3 4/4\nf 5/1 7/3 6/2\nf 5/1 8/4 7/3\n"
                    "f 1/1 5/1 6/2\nf 1/1 6/2 2/2\nf 2/2 6/2 7/3\nf 2/2 7/3 3/3\n"
                    "f 3/3 7/3 8/4\nf 3/3 8/4 4/4\nf 4/4 8/4 5/1\nf 4/4 5/1 1/1\n")
        pref = f"../../escenarios/variantes/{NOMBRE}"
        xml_assets += [f'<mesh name="m_{nombre}" file="{pref}/meshes/{nombre}.obj"/>',
                       f'<texture name="t_{nombre}" type="2d" file="../../variantes/{NOMBRE}/assets/{nombre}.png"/>',
                       f'<material name="mt_{nombre}" texture="t_{nombre}" rgba="1 1 1 1" specular="0.2" '
                       f'shininess="0.3" reflectance="0.05"/>']
        xml_geoms.append(f'<geom name="vis_{nombre}" type="mesh" mesh="m_{nombre}" material="mt_{nombre}" group="0" '
                         f'contype="0" conaffinity="0"/>')
        danos.append(dict(nombre=nombre, tipo=tipo, x0=x0, x1=x1, y0=y0, y1=y1, foto=r["archivo"],
                          sala=salas[k] if 0 <= k < len(salas) else ""))
        puestos.append((cx, cy))
        usados.append(dict(lugar=f"piso {danos[-1]['sala']}", tipo=tipo, foto=r["archivo"]))
    ruta = os.path.join(carpeta, "escenario.xml")
    with open(ruta, encoding="utf-8") as f:
        xml = f.read()
    xml = xml.replace("  </asset>", "".join("    " + a + "\n" for a in xml_assets) + "  </asset>", 1)
    xml = xml.replace("<worldbody>", "<worldbody>\n    " + "\n    ".join(xml_geoms), 1)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(xml)
    with open(os.path.join(carpeta, "danos_piso.json"), "w", encoding="utf-8") as f:
        json.dump(danos, f, indent=2, ensure_ascii=False)
    return usados


def main():
    variantes = os.path.join(gc.RAIZ, "escenarios", "variantes")
    carpeta = os.path.join(variantes, NOMBRE)
    if os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
    shutil.copytree(os.path.join(variantes, ORIGEN), carpeta)
    ruta = os.path.join(carpeta, "escenario.xml")
    with open(ruta, encoding="utf-8") as f:
        xml = f.read().replace(f"/{ORIGEN}/", f"/{NOMBRE}/")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(xml)

    modelo, datos = gc.construir_modelo(f"variantes/{NOMBRE}")
    nombres = gc.nombres_geoms(modelo)
    gc.colocar_robot(modelo, datos, 0, 0, 0, 0, 100, datos.qpos.copy())
    rng = np.random.default_rng(7)
    por_tipo = recortes()
    usados = paredes(carpeta, modelo, datos, nombres, rng, por_tipo)
    usados += pisos(carpeta, modelo, datos, nombres, rng, por_tipo)
    with open(os.path.join(carpeta, "danos_fotos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["lugar", "tipo", "foto"])
        w.writeheader()
        w.writerows(usados)
    with open(os.path.join(carpeta, "variante.json"), encoding="utf-8") as f:
        variante = json.load(f)
    variante.update(nombre=NOMBRE, base=ORIGEN, danos="fotos reales del conjunto de clasificacion (danos_fotos.csv)")
    with open(os.path.join(carpeta, "variante.json"), "w", encoding="utf-8") as f:
        json.dump(variante, f, indent=2, ensure_ascii=False)
    print(f"{NOMBRE}: {sum(u['lugar'].startswith('pared') for u in usados)} daños de pared y "
          f"{sum(u['lugar'].startswith('piso') for u in usados)} de piso con fotos reales")


if __name__ == "__main__":
    main()
