"""
Detalles de una casa real para las casas de ProcTHOR (estilo realista).

  - Muebles con forma en lugar de cajas: camas con respaldo, colchon, acolchado
    y almohadas; sillones con base, respaldo, apoyabrazos y almohadones; mesas y
    sillas con patas; comodas con cajones y manijas; estanterias con libros;
    mesadas de cocina con puertas, bacha y anafe; heladera, inodoro, bañera,
    lavatorio, tachos, lamparas de pie y plantas.
  - Los objetos que ProcTHOR pone arriba de los muebles (televisor, notebook,
    libros, lamparas, plantas, adornos) y los cuadros de las paredes.
  - Zocalos, alfeizares de ventana, llaves de luz junto a las puertas,
    tomacorrientes y lamparas de techo redondas.
  - Una sombra suave en el piso debajo de cada mueble.
  - Piso de ceramica para baños y cocinas.

La orientacion de cada mueble sale de la rotacion de ProcTHOR: el frente del
objeto mira hacia +z local (en este simulador, (sin r, cos r) en x, y).

Todo es visual (contype 0): las colisiones siguen siendo las cajas de antes.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

TELAS = [(0.42, 0.44, 0.48), (0.55, 0.50, 0.44), (0.30, 0.36, 0.44), (0.62, 0.60, 0.56), (0.46, 0.30, 0.26),
         (0.36, 0.42, 0.34), (0.72, 0.68, 0.60), (0.25, 0.25, 0.27)]
COLCHAS = [(0.85, 0.84, 0.80), (0.55, 0.62, 0.72), (0.74, 0.66, 0.56), (0.62, 0.42, 0.40), (0.48, 0.55, 0.48),
           (0.90, 0.88, 0.84), (0.40, 0.42, 0.50)]
LIBROS = [(0.55, 0.12, 0.10), (0.12, 0.22, 0.45), (0.15, 0.35, 0.20), (0.80, 0.70, 0.40), (0.20, 0.20, 0.22),
          (0.85, 0.82, 0.75), (0.45, 0.25, 0.40), (0.70, 0.40, 0.15)]
ADORNOS = [(0.85, 0.83, 0.78), (0.25, 0.25, 0.28), (0.55, 0.35, 0.22), (0.40, 0.55, 0.62), (0.70, 0.65, 0.55),
           (0.60, 0.20, 0.18)]


def _ruido(w, h, escala, r):
    base = r.random((max(2, int(h / escala)), max(2, int(w / escala))))
    return np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255.0


# ------------------------------------------------------------------ texturas
def textura_ceramica(destino, rng, lado_m=2.5, px=1024):
    """Piso de ceramica: baldosas con junta, leve variacion de tono entre piezas y brillo."""
    n = int(rng.choice([4, 5, 6, 8]))
    base = np.array([[232, 228, 220], [210, 205, 196], [190, 186, 180], [226, 222, 214], [176, 168, 156],
                     [236, 236, 234]][rng.integers(6)], np.float32)
    paso = px / n
    img = np.empty((px, px, 3), np.float32)
    yy, xx = np.mgrid[0:px, 0:px]
    fila, col = (yy // paso).astype(int), (xx // paso).astype(int)
    tonos = 1 + rng.normal(0, 0.025, (n, n))
    img[:] = base * tonos[fila % n, col % n][..., None]
    veta = gaussian_filter(rng.random((px, px)), 5, mode="wrap")
    veta = (veta - veta.mean()) / veta.std()
    img *= (1 + 0.025 * veta)[..., None]
    junta = max(2, int(px / lado_m * 0.004))
    borde = ((yy % paso) < junta) | ((xx % paso) < junta)
    img[borde] = base * 0.62
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(destino)


def textura_granito(destino, rng, px=256):
    base = np.array([[60, 58, 56], [205, 200, 192], [120, 112, 104], [30, 30, 32]][rng.integers(4)], np.float32)
    g = rng.random((px, px))
    img = base[None, None, :] * (0.85 + 0.3 * gaussian_filter(g, 0.8, mode="wrap"))[..., None]
    motas = rng.random((px, px)) > 0.93
    img[motas] = img[motas] * 0.5 + 120
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(destino)


def textura_cuadro(destino, rng, w=256, h=192):
    """Cuadro: paisaje simple, campos de color o figuras geometricas."""
    tipo = rng.integers(3)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    if tipo == 0:
        cielo = np.array([rng.uniform(120, 200), rng.uniform(150, 210), rng.uniform(190, 240)])
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=tuple(int(c) for c in cielo * (1 - 0.3 * t) + 60 * t))
        for k in range(3):
            base = h * (0.45 + 0.15 * k)
            verde = np.array([rng.uniform(40, 110), rng.uniform(80, 150), rng.uniform(40, 90)]) * (1 - 0.15 * k)
            pts = [(0, h)] + [(x, base - 25 * math.sin(x / w * math.pi * rng.uniform(1, 3) + k)) for x in range(0, w + 8, 8)] + [(w, h)]
            d.polygon(pts, fill=tuple(int(c) for c in verde))
    elif tipo == 1:
        colores = [tuple(int(c) for c in rng.uniform(30, 230, 3)) for _ in range(4)]
        y = 0
        for c in colores:
            alto = int(rng.uniform(0.15, 0.4) * h)
            d.rectangle([int(w * 0.08), y + 6, int(w * 0.92), y + alto], fill=c)
            y += alto
        img = img.filter(ImageFilter.GaussianBlur(2))
    else:
        d.rectangle([0, 0, w, h], fill=tuple(int(c) for c in rng.uniform(200, 245, 3)))
        for _ in range(int(rng.integers(4, 9))):
            c = tuple(int(v) for v in rng.uniform(20, 220, 3))
            x0, y0 = rng.uniform(0, w * 0.8), rng.uniform(0, h * 0.8)
            s = rng.uniform(20, 80)
            (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x0 + s, y0 + s * rng.uniform(0.5, 1.5)], fill=c)
    a = np.asarray(img, np.float32) * (0.92 + 0.08 * _ruido(w, h, 3, rng))[..., None]
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(destino)


# ------------------------------------------------------------------ geometria
class Detalles:
    def __init__(self, extra, carpeta_assets, pref, rng):
        self.extra = extra
        self.dir = carpeta_assets
        self.pref = pref
        self.rng = rng
        self.assets = []
        self.n = 0
        self.zonas_dano = []          # cajas del mundo que no hay que tapar
        r = rng
        self.mat("md_madera", texture="t_mueble", texuniform="true", texrepeat="1.5 1.5", rgba="0.82 0.78 0.74 1", specular="0.25", shininess="0.35")
        self.mat("md_madera_osc", texture="t_mueble", texuniform="true", texrepeat="1.5 1.5", rgba="0.55 0.48 0.42 1", specular="0.25", shininess="0.35")
        self.mat("md_blanco", rgba="0.92 0.92 0.90 1", specular="0.3", shininess="0.4")
        self.mat("md_ceramica", rgba="0.96 0.96 0.95 1", specular="0.6", shininess="0.8")
        self.mat("md_metal", rgba="0.72 0.73 0.75 1", specular="0.9", shininess="0.8")
        self.mat("md_negro", rgba="0.05 0.05 0.06 1", specular="0.7", shininess="0.9")
        self.mat("md_gris", rgba="0.32 0.33 0.35 1", specular="0.3", shininess="0.4")
        self.mat("md_lampara", rgba="1 0.97 0.88 1", emission="0.9")
        self.mat("md_pantalla", rgba="0.93 0.88 0.78 1", emission="0.35")
        self.mat("md_hoja", rgba="0.20 0.38 0.16 1", specular="0.2", shininess="0.3")
        self.mat("md_hoja2", rgba="0.28 0.46 0.20 1", specular="0.2", shininess="0.3")
        self.mat("md_maceta", rgba="0.58 0.32 0.20 1", specular="0.1", shininess="0.1")
        self.mat("md_espejo", rgba="0.78 0.82 0.86 1", specular="1", shininess="1", reflectance="0.3")
        self.mat("md_carton", rgba="0.62 0.48 0.32 1", specular="0.05", shininess="0.05")
        self.mat("md_sombra", rgba="0 0 0 0.11")
        textura_granito(os.path.join(carpeta_assets, "granito.png"), r)
        self.tex("td_granito", "granito.png")
        self.mat("md_mesada", texture="td_granito", texuniform="true", texrepeat="4 4", rgba="1 1 1 1",
                 specular="0.6", shininess="0.7")
        self.tex("td_tela", "tela.png")
        for i, (a, b, c) in enumerate(TELAS):
            self.mat(f"md_tela_{i}", texture="td_tela", texuniform="true", texrepeat="6 6", rgba=f"{a} {b} {c} 1",
                     specular="0.05", shininess="0.05")
        for i, (a, b, c) in enumerate(COLCHAS):
            self.mat(f"md_colcha_{i}", texture="td_tela", texuniform="true", texrepeat="5 5", rgba=f"{a} {b} {c} 1",
                     specular="0.05", shininess="0.05")
        self.mat("md_sabana", texture="td_tela", texuniform="true", texrepeat="5 5", rgba="0.95 0.95 0.93 1",
                 specular="0.05", shininess="0.05")
        for i, (a, b, c) in enumerate(LIBROS):
            self.mat(f"md_libro_{i}", rgba=f"{a} {b} {c} 1", specular="0.2", shininess="0.3")
        for i, (a, b, c) in enumerate(ADORNOS):
            self.mat(f"md_adorno_{i}", rgba=f"{a} {b} {c} 1", specular="0.4", shininess="0.5")
        self.cuadros = []
        for i in range(6):
            textura_cuadro(os.path.join(carpeta_assets, f"cuadro_{i}.png"), r)
            self.tex(f"td_cuadro_{i}", f"cuadro_{i}.png")
            self.mat(f"md_cuadro_{i}", texture=f"td_cuadro_{i}", rgba="1 1 1 1", specular="0.2", shininess="0.3")
            self.cuadros.append(f"md_cuadro_{i}")
        self.zocalo = "md_blanco" if r.random() < 0.6 else "md_madera"
        self.mueble_cocina = "md_blanco" if r.random() < 0.5 else "md_madera"

    # -------------------------------------------------- xml
    def tex(self, nombre, archivo):
        self.assets.append(f'<texture name="{nombre}" type="2d" file="{self.pref}/assets/{archivo}"/>')

    def mat(self, nombre, **kw):
        self.assets.append(f'<material name="{nombre}" ' + " ".join(f'{k}="{v}"' for k, v in kw.items()) + "/>")

    def _g(self, tipo, pos, size, mat, extra=""):
        self.n += 1
        self.extra.append(f'<geom name="det_{self.n}" type="{tipo}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
                          f'size="{" ".join(f"{s:.4f}" for s in size)}" material="{mat}" {extra}'
                          f'contype="0" conaffinity="0"/>')

    def caja(self, x0, x1, y0, y1, z0, z1, mat, extra=""):
        if x1 - x0 < 1e-3 or y1 - y0 < 1e-3 or z1 - z0 < 1e-3:
            return
        self._g("box", ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
                mat, extra)

    def cilindro(self, cx, cy, z0, z1, radio, mat, extra=""):
        self._g("cylinder", (cx, cy, (z0 + z1) / 2), (radio, (z1 - z0) / 2), mat, extra)

    def elipsoide(self, cx, cy, cz, rx, ry, rz, mat):
        self._g("ellipsoid", (cx, cy, cz), (rx, ry, rz), mat)

    def inyectar(self, ruta_xml):
        with open(ruta_xml, encoding="utf-8") as f:
            xml = f.read()
        xml = xml.replace("  </asset>", "".join("    " + a + "\n" for a in self.assets) + "  </asset>", 1)
        with open(ruta_xml, "w", encoding="utf-8") as f:
            f.write(xml)

    def tapa_dano(self, x0, x1, y0, y1, z0, z1):
        return any(x0 < b[1] and b[0] < x1 and y0 < b[3] and b[2] < y1 and z0 < b[5] and b[4] < z1
                   for b in self.zonas_dano)

    # -------------------------------------------------- marco local de un mueble
    def local(self, x0, x1, y0, y1, z0, frente):
        """Devuelve (ancho, profundidad, b) donde b(u0, u1, v0, v1, z0, z1, mat) dibuja una caja en
        coordenadas del mueble: u a lo ancho, v desde atras (0) hacia el frente, z desde su base."""
        if frente in ("+y", "-y"):
            W, D = x1 - x0, y1 - y0
        else:
            W, D = y1 - y0, x1 - x0

        def a_mundo(u, v):
            if frente == "+y":
                return x0 + u, y0 + v
            if frente == "-y":
                return x1 - u, y1 - v
            if frente == "+x":
                return x0 + v, y1 - u
            return x1 - v, y0 + u

        def b(u0, u1, v0, v1, zz0, zz1, mat):
            (ax, ay), (bx, by) = a_mundo(u0, v0), a_mundo(u1, v1)
            self.caja(min(ax, bx), max(ax, bx), min(ay, by), max(ay, by), z0 + zz0, z0 + zz1, mat)

        def e(u, v, zz, ru, rv, rz, mat):
            px, py = a_mundo(u, v)
            rx, ry = (ru, rv) if frente in ("+y", "-y") else (rv, ru)
            self.elipsoide(px, py, z0 + zz, rx, ry, rz, mat)

        def c(u, v, zz0, zz1, radio, mat):
            px, py = a_mundo(u, v)
            self.cilindro(px, py, z0 + zz0, z0 + zz1, radio, mat)
        b.e, b.c = e, c
        return W, D, b

    def sombra(self, x0, x1, y0, y1):
        for k, m in enumerate((0.02, 0.07, 0.13)):
            z = 0.0046 + 0.0004 * k
            self.caja(x0 - m, x1 + m, y0 - m, y1 + m, z, z + 0.0003, "md_sombra")

    # -------------------------------------------------- muebles de piso
    def mueble(self, caja, tipo, frente):
        x0, x1, y0, y1, z0, z1 = caja
        Hm = z1 - z0
        W, D, b = self.local(x0, x1, y0, y1, z0, frente)
        r = self.rng
        self.sombra(x0, x1, y0, y1)
        if tipo == "Bed":
            alto_cab = float(np.clip(Hm, 0.8, 1.25))
            b(0, W, 0.05, D, 0.06, 0.30, "md_madera")
            b(0, W, 0, 0.06, 0, alto_cab, "md_madera")
            b(0.03, W - 0.03, 0.07, D - 0.02, 0.30, 0.50, "md_sabana")
            colcha = f"md_colcha_{r.integers(len(COLCHAS))}"
            b(-0.02, W + 0.02, 0.38 * D, D + 0.02, 0.34, 0.54, colcha)
            n = 2 if W > 1.2 else 1
            pw = (W - 0.16) / n
            for i in range(n):
                b.e(0.08 + pw * (i + 0.5), 0.24, 0.58, pw * 0.45, 0.14, 0.07, "md_sabana")
        elif tipo in ("Sofa", "ArmChair"):
            tela = f"md_tela_{r.integers(len(TELAS))}"
            for u in (0.06, W - 0.06):
                for v in (0.06, D - 0.06):
                    b(u - 0.025, u + 0.025, v - 0.025, v + 0.025, 0, 0.08, "md_madera_osc")
            b(0, W, 0, D, 0.08, 0.38, tela)
            alto = float(np.clip(Hm, 0.7, 1.0))
            b(0, W, 0, 0.2, 0.38, alto, tela)
            brazo = 0.15 if W > 1.0 else 0.12
            b(0, brazo, 0, D, 0.38, 0.62, tela)
            b(W - brazo, W, 0, D, 0.38, 0.62, tela)
            n = max(1, int(round((W - 2 * brazo) / 0.62)))
            cw = (W - 2 * brazo) / n
            for i in range(n):
                ua = brazo + cw * i
                b(ua + 0.008, ua + cw - 0.008, 0.2, D - 0.01, 0.38, 0.49, tela)
                b(ua + 0.01, ua + cw - 0.01, 0.2, 0.33, 0.49, alto - 0.04, tela)
            if tipo == "Sofa" and r.random() < 0.7:
                otra = f"md_tela_{r.integers(len(TELAS))}"
                b.e(brazo + 0.22, 0.4, 0.64, 0.18, 0.07, 0.16, otra)
                b.e(W - brazo - 0.22, 0.4, 0.64, 0.18, 0.07, 0.16, otra)
        elif tipo in ("DiningTable", "CoffeeTable", "SideTable", "Desk"):
            mat = "md_madera" if r.random() < 0.75 else "md_blanco"
            b(0, W, 0, D, Hm - 0.035, Hm, mat)
            p = 0.045
            for u in (0.03, W - 0.03 - p):
                for v in (0.03, D - 0.03 - p):
                    b(u, u + p, v, v + p, 0, Hm - 0.035, mat)
            if tipo == "Desk" and W > 0.9:
                b(W - 0.42, W - 0.02, 0.02, D - 0.02, Hm - 0.6, Hm - 0.035, mat)
                for k in range(3):
                    zz = Hm - 0.6 + 0.19 * k
                    b(W - 0.41, W - 0.03, D - 0.02, D - 0.008, zz + 0.01, zz + 0.18, mat)
                    b(W - 0.25, W - 0.19, D - 0.008, D + 0.01, zz + 0.09, zz + 0.105, "md_metal")
            if tipo == "CoffeeTable":
                b(0.04, W - 0.04, 0.04, D - 0.04, 0.12, 0.14, mat)
        elif tipo in ("Chair", "Stool"):
            mat = "md_madera" if r.random() < 0.8 else "md_negro"
            asiento = min(0.46, Hm * 0.6) if tipo == "Chair" else Hm
            b(0, W, 0, D, asiento - 0.035, asiento, mat)
            p = 0.035
            for u in (0.02, W - 0.02 - p):
                for v in (0.02, D - 0.02 - p):
                    b(u, u + p, v, v + p, 0, asiento - 0.035, mat)
            if tipo == "Chair":
                b(0.02, 0.02 + p, 0.02, 0.02 + p, asiento, Hm, mat)
                b(W - 0.02 - p, W - 0.02, 0.02, 0.02 + p, asiento, Hm, mat)
                b(0.02, W - 0.02, 0.02, 0.05, Hm - 0.18, Hm, mat)
        elif tipo in ("Dresser", "TVStand", "Cabinet", "Drawer"):
            mat = "md_madera" if r.random() < 0.7 else "md_blanco"
            frente_mat = mat if r.random() < 0.6 else ("md_madera_osc" if mat == "md_madera" else "md_gris")
            b(0.03, W - 0.03, 0.03, D - 0.03, 0, 0.06, "md_negro")
            b(0, W, 0, D - 0.015, 0.06, Hm, mat)
            filas = int(np.clip(round((Hm - 0.06) / 0.24), 1, 5))
            cols = 2 if W > 1.0 else 1
            alto = (Hm - 0.08) / filas
            ancho = (W - 0.04) / cols
            for f in range(filas):
                for c in range(cols):
                    ua, za = 0.02 + ancho * c, 0.07 + alto * f
                    b(ua + 0.006, ua + ancho - 0.006, D - 0.015, D, za + 0.006, za + alto - 0.006, frente_mat)
                    um, zm = ua + ancho / 2, za + alto * 0.62
                    b(um - 0.06, um + 0.06, D, D + 0.018, zm - 0.008, zm + 0.008, "md_metal")
        elif tipo in ("ShelvingUnit", "Shelf", "Bookcase"):
            mat = "md_madera" if r.random() < 0.7 else "md_blanco"
            e = 0.025
            b(0, e, 0, D, 0, Hm, mat)
            b(W - e, W, 0, D, 0, Hm, mat)
            b(0, W, 0, 0.012, 0, Hm, mat)
            n = max(2, int(Hm / 0.36))
            for k in range(n + 1):
                zz = min(Hm - e, k * (Hm - e) / n)
                b(e, W - e, 0, D, zz, zz + e, mat)
                if k < n and r.random() < 0.85:
                    self._libros(b, e, W - e, 0.03, D - 0.02, zz + e, (Hm - e) / n - e - 0.02)
        elif tipo == "CounterTop":
            b(0.0, W, 0.0, D - 0.07, 0, 0.1, "md_negro")
            b(0, W, 0, D - 0.02, 0.1, Hm - 0.035, self.mueble_cocina)
            b(-0.01, W + 0.01, 0, D + 0.02, Hm - 0.035, Hm, "md_mesada")
            n = max(1, int(round(W / 0.55)))
            pw = W / n
            for i in range(n):
                ua = pw * i
                b(ua + 0.004, ua + pw - 0.004, D - 0.02, D - 0.005, Hm - 0.2, Hm - 0.04, self.mueble_cocina)
                b(ua + 0.004, ua + pw - 0.004, D - 0.02, D - 0.005, 0.11, Hm - 0.205, self.mueble_cocina)
                b(ua + pw / 2 - 0.07, ua + pw / 2 + 0.07, D - 0.005, D + 0.012, Hm - 0.13, Hm - 0.115, "md_metal")
                b(ua + pw - 0.06, ua + pw - 0.045, D - 0.005, D + 0.012, Hm - 0.45, Hm - 0.3, "md_metal")
            if W > 1.2:
                us = W * r.uniform(0.25, 0.4)
                b(us - 0.3, us + 0.3, D * 0.25, D * 0.8, Hm - 0.004, Hm + 0.002, "md_metal")
                b(us - 0.26, us + 0.26, D * 0.29, D * 0.76, Hm - 0.002, Hm + 0.003, "md_gris")
                b.c(us, D * 0.14, Hm, Hm + 0.3, 0.015, "md_metal")
                b(us - 0.012, us + 0.012, D * 0.14, D * 0.32, Hm + 0.27, Hm + 0.3, "md_metal")
                ua = W * r.uniform(0.6, 0.75)
                if ua + 0.3 < W:
                    b(ua - 0.3, ua + 0.3, D * 0.2, D * 0.85, Hm, Hm + 0.006, "md_negro")
                    for du in (-0.15, 0.15):
                        for dv in (-0.15, 0.15):
                            b.c(ua + du, D * 0.52 + dv, Hm + 0.006, Hm + 0.008, 0.08, "md_gris")
        elif tipo == "Fridge":
            mat = "md_blanco" if r.random() < 0.5 else "md_metal"
            b(0, W, 0, D - 0.04, 0, Hm, mat)
            corte = Hm * 0.64
            b(0.005, W - 0.005, D - 0.04, D, 0.005, corte - 0.005, mat)
            b(0.005, W - 0.005, D - 0.04, D, corte + 0.005, Hm - 0.005, mat)
            b(W - 0.09, W - 0.06, D, D + 0.04, corte - 0.55, corte - 0.1, "md_gris")
            b(W - 0.09, W - 0.06, D, D + 0.04, corte + 0.1, corte + 0.4, "md_gris")
        elif tipo == "Toilet":
            b(W * 0.3, W * 0.7, 0.18, D * 0.72, 0, 0.36, "md_ceramica")
            b.e(W / 2, D * 0.6, 0.37, W * 0.45, D * 0.36, 0.07, "md_ceramica")
            b.e(W / 2, D * 0.6, 0.43, W * 0.44, D * 0.35, 0.015, "md_ceramica")
            b(0.02, W - 0.02, 0, 0.2, 0.4, min(Hm, 0.82), "md_ceramica")
            b(W * 0.45, W * 0.55, 0.06, 0.12, min(Hm, 0.82), min(Hm, 0.82) + 0.01, "md_metal")
        elif tipo == "Bathtub":
            e = 0.07
            b(0, W, 0, D, 0, 0.1, "md_ceramica")
            b(0, W, 0, e, 0.1, Hm, "md_ceramica")
            b(0, W, D - e, D, 0.1, Hm, "md_ceramica")
            b(0, e, 0, D, 0.1, Hm, "md_ceramica")
            b(W - e, W, 0, D, 0.1, Hm, "md_ceramica")
            b.c(W - 0.15, 0.04, Hm, Hm + 0.15, 0.015, "md_metal")
        elif tipo == "Sink":
            alto = float(np.clip(Hm, 0.75, 0.95))
            b(W * 0.36, W * 0.64, 0.03, 0.24, 0, alto - 0.15, "md_ceramica")
            b(0, W, 0, D, alto - 0.15, alto, "md_ceramica")
            b(0.05, W - 0.05, 0.12, D - 0.05, alto - 0.004, alto + 0.001, "md_gris")
            b.c(W / 2, 0.06, alto, alto + 0.2, 0.014, "md_metal")
            b(W / 2 - 0.012, W / 2 + 0.012, 0.06, 0.18, alto + 0.17, alto + 0.2, "md_metal")
        elif tipo in ("GarbageCan", "LaundryHamper"):
            mat = "md_gris" if tipo == "GarbageCan" else "md_carton"
            radio = min(W, D) / 2
            b.c(W / 2, D / 2, 0, Hm, radio, mat)
            b.c(W / 2, D / 2, Hm, Hm + 0.012, radio * 1.03, "md_negro" if tipo == "GarbageCan" else mat)
        elif tipo == "FloorLamp":
            b.c(W / 2, D / 2, 0, 0.025, 0.14, "md_negro")
            b.c(W / 2, D / 2, 0.025, Hm - 0.2, 0.012, "md_metal")
            b.c(W / 2, D / 2, Hm - 0.32, Hm, 0.2, "md_pantalla")
        elif tipo == "HousePlant":
            self._planta(b, W / 2, D / 2, 0, Hm, min(W, D) / 2)
        elif tipo in ("Ottoman", "DogBed"):
            tela = f"md_tela_{r.integers(len(TELAS))}"
            if tipo == "DogBed":
                b.e(W / 2, D / 2, Hm / 2, W / 2, D / 2, Hm / 2, tela)
            else:
                b(0, W, 0, D, 0.05, Hm, tela)
                b(0.04, W - 0.04, 0.04, D - 0.04, 0, 0.05, "md_madera_osc")
        elif tipo == "GarbageBag":
            b.e(W / 2, D / 2, Hm / 2, W / 2, D / 2, Hm / 2, "md_negro")
        elif tipo == "BasketBall":
            b.e(W / 2, D / 2, Hm / 2, W / 2, D / 2, Hm / 2, "md_adorno_5")
        elif tipo == "Box":
            b(0, W, 0, D, 0, Hm, "md_carton")
        elif tipo == "Safe":
            b(0, W, 0, D, 0, Hm, "md_gris")
            b.c(W * 0.7, D, Hm * 0.55, Hm * 0.55 + 0.001, 0.04, "md_metal")
        else:
            b(0, W, 0, D, 0, Hm, "md_gris")

    def _libros(self, b, u0, u1, v0, v1, z0, alto_max):
        r = self.rng
        u = u0 + r.uniform(0, 0.08)
        while u < u1 - 0.05:
            if r.random() < 0.15:
                u += r.uniform(0.08, 0.3)
                continue
            esp = r.uniform(0.02, 0.05)
            alto = min(alto_max, r.uniform(0.16, 0.30))
            prof = min(v1 - v0, r.uniform(0.14, 0.22))
            if u + esp > u1:
                break
            b(u, u + esp, v1 - prof, v1, z0, z0 + alto, f"md_libro_{r.integers(len(LIBROS))}")
            u += esp + 0.002

    def _planta(self, b, u, v, z0, alto, radio):
        r = self.rng
        maceta = min(0.4, alto * 0.35)
        b.c(u, v, z0, z0 + maceta, radio * 0.6, "md_maceta")
        for _ in range(int(r.integers(4, 8))):
            du, dv = r.normal(0, radio * 0.35, 2)
            zz = z0 + maceta + r.uniform(0.1, 0.8) * (alto - maceta)
            s = r.uniform(0.3, 0.6) * radio
            b.e(u + du, v + dv, zz, s, s * r.uniform(0.7, 1.2), s * r.uniform(0.6, 1.4),
                "md_hoja" if r.random() < 0.5 else "md_hoja2")

    # -------------------------------------------------- objetos sobre muebles y paredes
    def objeto(self, caja, tipo, frente):
        x0, x1, y0, y1, z0, z1 = caja
        Hm = z1 - z0
        W, D, b = self.local(x0, x1, y0, y1, z0, frente)
        r = self.rng
        if tipo == "Television":
            b(0, W, max(0, D / 2 - 0.03), min(D, D / 2 + 0.03), 0.08, Hm, "md_negro")
            b(W / 2 - 0.12, W / 2 + 0.12, D / 2 - 0.12, D / 2 + 0.12, 0, 0.02, "md_negro")
            b(W / 2 - 0.03, W / 2 + 0.03, D / 2 - 0.03, D / 2, 0.02, 0.1, "md_negro")
        elif tipo == "Laptop":
            b(0, W, 0, D, 0, 0.02, "md_gris")
            b(0, W, 0, 0.015, 0.02, min(Hm, 0.22), "md_gris")
        elif tipo in ("DeskLamp", "TableLamp"):
            b.c(W / 2, D / 2, 0, 0.02, min(W, D) / 2 * 0.8, "md_negro")
            b.c(W / 2, D / 2, 0.02, Hm * 0.65, 0.008, "md_metal")
            b.c(W / 2, D / 2, Hm * 0.55, Hm, min(W, D) / 2, "md_pantalla")
        elif tipo == "HousePlant":
            self._planta(b, W / 2, D / 2, 0, Hm, min(W, D) / 2)
        elif tipo == "Book":
            b(0, W, 0, D, 0, Hm, f"md_libro_{r.integers(len(LIBROS))}")
        elif tipo in ("Bowl", "Pot", "Vase", "Mug", "Cup", "Bottle", "WineBottle", "Candle", "Statue",
                      "SoapBottle", "Kettle", "CoffeeMachine", "Toaster", "TissueBox", "AlarmClock", "Pan"):
            mat = f"md_adorno_{r.integers(len(ADORNOS))}"
            if tipo in ("TissueBox", "CoffeeMachine", "Toaster", "AlarmClock"):
                b(0, W, 0, D, 0, Hm, mat)
            else:
                b.c(W / 2, D / 2, 0, Hm, min(W, D) / 2, mat)
        elif tipo == "Pillow":
            b.e(W / 2, D / 2, Hm / 2, W / 2, D / 2, Hm / 2, f"md_tela_{r.integers(len(TELAS))}")
        else:
            b(0, W, 0, D, 0, Hm, f"md_adorno_{r.integers(len(ADORNOS))}")

    def cuadro(self, x0, x1, y0, y1, z0, z1, normal):
        """Cuadro colgado: marco y tela pintada. normal: 'x' o 'y' (eje perpendicular a la pared)."""
        if self.tapa_dano(x0 - 0.05, x1 + 0.05, y0 - 0.05, y1 + 0.05, z0 - 0.05, z1 + 0.05):
            return
        r = self.rng
        marco = ["md_negro", "md_madera", "md_blanco", "md_madera_osc"][r.integers(4)]
        tela = self.cuadros[r.integers(len(self.cuadros))]
        self.caja(x0, x1, y0, y1, z0, z1, marco)
        m = 0.035
        if normal == "x":
            self.caja(x0 - 0.002, x1 + 0.002, y0 + m, y1 - m, z0 + m, z1 - m, tela)
        else:
            self.caja(x0 + m, x1 - m, y0 - 0.002, y1 + 0.002, z0 + m, z1 - m, tela)

    def espejo(self, x0, x1, y0, y1, z0, z1, normal):
        if self.tapa_dano(x0 - 0.05, x1 + 0.05, y0 - 0.05, y1 + 0.05, z0 - 0.05, z1 + 0.05):
            return
        self.caja(x0, x1, y0, y1, z0, z1, "md_metal")
        m = 0.02
        if normal == "x":
            self.caja(x0 - 0.002, x1 + 0.002, y0 + m, y1 - m, z0 + m, z1 - m, "md_espejo")
        else:
            self.caja(x0 + m, x1 - m, y0 - 0.002, y1 + 0.002, z0 + m, z1 - m, "md_espejo")

    # -------------------------------------------------- terminaciones de la casa
    def zocalos(self, eje, pos, a0, a1, aberturas, T, slab):
        """Zocalo de 8 cm a cada lado de la pared, cortado en las puertas."""
        e, alto = 0.012, 0.08
        tramos, cur = [], a0
        for c, w, z0, z1, tipo in sorted(aberturas):
            if tipo == "puerta":
                tramos.append((cur, c - w / 2 - 0.05))
                cur = c + w / 2 + 0.05
        tramos.append((cur, a1))
        for s0, s1 in tramos:
            if s1 - s0 < 0.05:
                continue
            for lado in (-1, 1):
                off = pos + lado * (T / 2 + e / 2)
                if eje == "x":
                    self.caja(s0, s1, off - e / 2, off + e / 2, slab, alto, self.zocalo)
                else:
                    self.caja(off - e / 2, off + e / 2, s0, s1, slab, alto, self.zocalo)

    def alfeizar(self, eje, pos, c, w, z0, T):
        for lado in (-1, 1):
            a, b = pos + lado * T / 2, pos + lado * (T / 2 + 0.05)
            lo, hi = min(a, b), max(a, b)
            if eje == "x":
                self.caja(c - w / 2 - 0.05, c + w / 2 + 0.05, lo, hi, z0 - 0.03, z0, "md_blanco")
            else:
                self.caja(lo, hi, c - w / 2 - 0.05, c + w / 2 + 0.05, z0 - 0.03, z0, "md_blanco")

    def placa(self, eje, pos, s, z, lado, T, ancho=0.08, alto=0.12, tecla=True):
        """Llave de luz o tomacorriente sobre la cara de la pared que da hacia 'lado' (+1/-1)."""
        e = 0.008
        cara = pos + lado * T / 2
        lo, hi = sorted((cara, cara + lado * e))
        if eje == "x":
            caja = (s - ancho / 2, s + ancho / 2, lo, hi, z - alto / 2, z + alto / 2)
        else:
            caja = (lo, hi, s - ancho / 2, s + ancho / 2, z - alto / 2, z + alto / 2)
        if self.tapa_dano(*caja):
            return
        self.caja(*caja, "md_blanco")
        lo2, hi2 = sorted((cara + lado * e, cara + lado * (e + 0.006)))
        if tecla:
            if eje == "x":
                self.caja(s - 0.018, s + 0.018, lo2, hi2, z - 0.025, z + 0.025, "md_blanco")
            else:
                self.caja(lo2, hi2, s - 0.018, s + 0.018, z - 0.025, z + 0.025, "md_blanco")
        else:
            for dz in (-0.015, 0.015):
                if eje == "x":
                    self.caja(s - 0.012, s - 0.006, lo2, hi2, z + dz - 0.006, z + dz + 0.006, "md_negro")
                    self.caja(s + 0.006, s + 0.012, lo2, hi2, z + dz - 0.006, z + dz + 0.006, "md_negro")

    def lampara_techo(self, cx, cy, H):
        self.cilindro(cx, cy, H - 0.06, H - 0.002, 0.03, "md_blanco", 'group="1" ')
        self.cilindro(cx, cy, H - 0.14, H - 0.06, 0.2, "md_lampara", 'group="1" ')
