"""
Genera el escenario "edificio_informe": 4 salas + vestibulo central, con daño,
pensado para capturas de calidad en el informe.

Salida (todo dentro de esta carpeta):
  escenario.xml          -> <mujocoinclude> compatible con launch_g1_nav.py
  meshes/*.obj           -> geometria visual con UV (una malla por material)
  assets/decal_*.png     -> grietas / manchas / desprendimientos (RGB, opacos)
  damage_manifest.json   -> lista de daños con posicion y tipo

Requiere las texturas de assets/ (python descargar_texturas.py).
Uso:  python generar_edificio.py
La colision son cajas simples en group=3 (invisibles); lo visual es malla
con contype=0, igual que el resto de escenarios del repo.

Nota sobre los daños de superficie: MuJoCo 3.11 ignora el canal alfa de las
texturas, asi que cada grieta/mancha se "hornea": se toma la misma textura de
la pared, en las mismas coordenadas del mundo, y se le compone el daño encima.
El resultado es opaco y empalma con la pared sin costura.
"""
import json
import math
import os

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull

AQUI = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(AQUI, "assets")
MESHES = os.path.join(AQUI, "meshes")
os.makedirs(MESHES, exist_ok=True)

# Prefijo relativo desde unitree_g1/assets (meshdir de g1.xml), igual que habitacion_daniada
PREF = "../../escenarios/edificio_informe"

RNG = np.random.default_rng(7)

# ------------------------------------------------------------------ parametros
H = 2.8          # altura de pared
T = 0.20         # espesor de pared
X0, X1, Y0, Y1 = -8.0, 8.0, -3.0, 9.0
PUERTA_W, PUERTA_H = 1.0, 2.1
VENT_W, VENT_Z0, VENT_Z1 = 1.4, 0.9, 2.1
TECHO = False    # sin techo: se ve mejor el interior desde arriba y en las capturas
SLAB = 0.004     # el piso visual queda 4 mm sobre el plano de colision
TILE_REVOQUE = 2.0
PX_POR_M = 420   # resolucion de los decals horneados


# ------------------------------------------------------------------ mallas
class Malla:
    """Acumula cajas/poligonos con UV en metros (tile = lado de una repeticion)."""

    def __init__(self, tile=2.0):
        self.v, self.vt, self.vn, self.f = [], [], [], []
        self.tile = tile

    def _quad(self, pts, n, uvs):
        pts = [np.asarray(p, float) for p in pts]
        n = np.asarray(n, float)
        if np.dot(np.cross(pts[1] - pts[0], pts[2] - pts[0]), n) < 0:
            pts, uvs = pts[::-1], uvs[::-1]
        b = len(self.v)
        for p, uv in zip(pts, uvs):
            self.v.append(p)
            self.vt.append(uv)
            self.vn.append(n)
        for a, c, d in ((0, 1, 2), (0, 2, 3)):
            self.f.append((b + a, b + c, b + d))

    def caja(self, x0, x1, y0, y1, z0, z1, tile=None, caras="xXyYzZ"):
        t = tile or self.tile
        c = {
            "x": ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0), lambda p: (p[1] / t, p[2] / t)),
            "X": ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0), lambda p: (p[1] / t, p[2] / t)),
            "y": ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0), lambda p: (p[0] / t, p[2] / t)),
            "Y": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0), lambda p: (p[0] / t, p[2] / t)),
            "z": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1), lambda p: (p[0] / t, p[1] / t)),
            "Z": ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1), lambda p: (p[0] / t, p[1] / t)),
        }
        for k in caras:
            pts, n, uvf = c[k]
            self._quad(pts, n, [uvf(p) for p in pts])

    def poligono_convexo(self, puntos, tile=0.6):
        """Roca/escombro: casco convexo con UV triplanar."""
        h = ConvexHull(puntos)
        cen = puntos.mean(axis=0)
        for s in h.simplices:
            p = puntos[s]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            n /= np.linalg.norm(n) + 1e-12
            if np.dot(n, p[0] - cen) < 0:
                n = -n
            ax = int(np.argmax(np.abs(n)))
            idx = [i for i in range(3) if i != ax]
            uvs = [(q[idx[0]] / tile, q[idx[1]] / tile) for q in p]
            b = len(self.v)
            order = [0, 1, 2]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), n) < 0:
                order = [0, 2, 1]
            for i in order:
                self.v.append(p[i])
                self.vt.append(uvs[i])
                self.vn.append(n)
            self.f.append((b, b + 1, b + 2))

    def guardar(self, nombre):
        ruta = os.path.join(MESHES, nombre + ".obj")
        with open(ruta, "w") as fh:
            fh.write(f"# {nombre}\n")
            for p in self.v:
                fh.write("v %.5f %.5f %.5f\n" % tuple(p))
            for t in self.vt:
                fh.write("vt %.5f %.5f\n" % tuple(t))
            for n in self.vn:
                fh.write("vn %.4f %.4f %.4f\n" % tuple(n))
            for a, b, c in self.f:
                fh.write(f"f {a+1}/{a+1}/{a+1} {b+1}/{b+1}/{b+1} {c+1}/{c+1}/{c+1}\n")
        return nombre


# mallas por material
M = {k: Malla(t) for k, t in dict(
    revoque=2.0, hormigon=2.0, piso_h=2.5, piso_m=2.0, suelo=3.0, techo=2.0,
    madera=1.0, marco=1.0, luz=1.0, vidrio=1.0,
).items()}
COL = []          # cajas de colision: (nombre, cx, cy, cz, sx, sy, sz)
EXTRA_XML = []    # geoms de primitivas (cilindros, rebar, etc.)
MANIFEST = []


def col_caja(nombre, x0, x1, y0, y1, z0, z1):
    COL.append((nombre, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2))


def solido(mat, nombre, x0, x1, y0, y1, z0, z1, col=True, **kw):
    M[mat].caja(x0, x1, y0, y1, z0, z1, **kw)
    if col:
        col_caja(nombre, x0, x1, y0, y1, z0, z1)


# ------------------------------------------------------------------ paredes
PX_PIEZA = 380   # px por metro en las piezas de pared con daño horneado
PIEZAS_DANADAS = []   # (nombre_pieza, textura)


def pieza_danada(nombre, caja, eje, danos, tile=TILE_REVOQUE, base_png="pared_revoque.png"):
    """Pared con daño horneado: una malla propia + una textura atlas (cara A abajo, cara B arriba).
    La textura base es la misma de la pared, muestreada en coordenadas de mundo, asi que empalma sin costura."""
    x0, x1, y0, y1, z0, z1 = caja
    if eje == "y":      # la pared corre a lo largo de y; sus caras tienen normal -x (A) y +x (B)
        s0, s1, cara_a, cara_b = y0, y1, "x", "X"
    else:               # corre a lo largo de x; normales -y (A) y +y (B)
        s0, s1, cara_a, cara_b = x0, x1, "y", "Y"
    L, Hh = s1 - s0, z1 - z0
    Wp, Hp = max(8, int(L * PX_PIEZA)), max(8, int(Hh * PX_PIEZA))
    base = _base(base_png)
    S = base.shape[0]
    s = s0 + (np.arange(Wp) + 0.5) / Wp * L
    z = z1 - (np.arange(Hp) + 0.5) / Hp * Hh            # fila 0 = arriba de la pared
    col = ((s / tile) % 1.0 * S).astype(int) % S
    row = ((1.0 - (z / tile) % 1.0) * S).astype(int) % S
    cara_img = {c: base[row[:, None], col[None, :]].copy() for c in (cara_a, cara_b)}
    for d in danos:
        img = cara_img[d["cara"]]
        ov_rgb, ov_a = d["ov"](int(d["w"] * PX_PIEZA), int(d["h"] * PX_PIEZA))
        h_o, w_o = ov_a.shape
        c0 = int(round((d["s"] - d["w"] / 2 - s0) * PX_PIEZA))
        r0 = int(round((z1 - (d["z"] + d["h"] / 2)) * PX_PIEZA))
        ca, cb = max(0, c0), min(Wp, c0 + w_o)
        ra, rb = max(0, r0), min(Hp, r0 + h_o)
        if ca >= cb or ra >= rb:
            continue
        sub_a = ov_a[ra - r0:rb - r0, ca - c0:cb - c0][..., None]
        sub_rgb = ov_rgb[ra - r0:rb - r0, ca - c0:cb - c0]
        img[ra:rb, ca:cb] = img[ra:rb, ca:cb] * (1 - sub_a) + sub_rgb * sub_a
        MANIFEST.append(dict(tipo=d["tipo"], sala=d["sala"], pieza=nombre, cara=d["cara"],
                             centro_s=d["s"], centro_z=d["z"], ancho=d["w"], alto=d["h"]))
    atlas = np.vstack([cara_img[cara_b], cara_img[cara_a]])   # arriba: B (v 0.5..1), abajo: A (v 0..0.5)
    tex = f"pieza_{nombre}.png"
    Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8)).save(os.path.join(ASSETS, tex))

    m = Malla()

    def uv_de(cara, p):
        sc = p[1] if eje == "y" else p[0]
        u = (sc - s0) / L
        vl = (p[2] - z0) / Hh
        return (min(max(u, 0.002), 0.998), (0.0 if cara == cara_a else 0.5) + 0.5 * min(max(vl, 0.002), 0.998))

    caras = {
        "x": ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], (-1, 0, 0)),
        "X": ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0)),
        "y": ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0, -1, 0)),
        "Y": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0)),
        "z": ([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)], (0, 0, -1)),
        "Z": ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1)),
    }
    for k, (pts, n) in caras.items():
        principal = k in (cara_a, cara_b)
        ref = k if principal else cara_a
        m._quad(pts, n, [uv_de(ref, np.asarray(q, float)) for q in pts])
    m.guardar(f"pieza_{nombre}")
    PIEZAS_DANADAS.append((f"pieza_{nombre}", tex))
    col_caja(nombre, x0, x1, y0, y1, z0, z1)


def pared(nombre, eje, pos, a0, a1, aberturas=(), mat="revoque", danos=()):
    """eje 'x': corre a lo largo de x en y=pos.  aberturas: (centro, ancho, z0, z1, tipo).
    danos: dicts con cara ('x','X','y','Y'), s (coordenada a lo largo), z, w, h, ov (overlay), tipo, sala."""
    aberturas = sorted(aberturas)
    piezas = []  # (s0, s1, z0, z1)
    cur = a0
    for c, w, z0, z1, tipo in aberturas:
        s0, s1 = c - w / 2, c + w / 2
        if s0 > cur:
            piezas.append((cur, s0, 0, H))
        if z0 > 0:
            piezas.append((s0, s1, 0, z0))
        piezas.append((s0, s1, z1, H))
        cur = s1
    if a1 > cur:
        piezas.append((cur, a1, 0, H))
    for i, (s0, s1, z0, z1) in enumerate(piezas):
        if eje == "x":
            args = (s0, s1, pos - T / 2, pos + T / 2, z0, z1)
        else:
            args = (pos - T / 2, pos + T / 2, s0, s1, z0, z1)
        mis = [d for d in danos if s0 <= d["s"] <= s1 and z0 <= d["z"] <= z1]
        if mis:
            pieza_danada(f"{nombre}_{i}", args, eje, mis)
        else:
            solido(mat, f"{nombre}_{i}", *args)
    for c, w, z0, z1, tipo in aberturas:
        marco(nombre, eje, pos, c, w, z0, z1, tipo)


def marco(nombre, eje, pos, c, w, z0, z1, tipo):
    g, d = 0.05, T + 0.03
    s0, s1 = c - w / 2, c + w / 2

    def caja(a0_, a1_, zz0, zz1):
        if eje == "x":
            M["marco"].caja(a0_, a1_, pos - d / 2, pos + d / 2, zz0, zz1)
        else:
            M["marco"].caja(pos - d / 2, pos + d / 2, a0_, a1_, zz0, zz1)

    caja(s0, s0 + g, z0, z1)
    caja(s1 - g, s1, z0, z1)
    caja(s0, s1, z1 - g, z1)
    if tipo == "ventana":
        caja(s0, s1, z0, z0 + g)
        if eje == "x":
            M["vidrio"].caja(s0, s1, pos - 0.006, pos + 0.006, z0, z1)
        else:
            M["vidrio"].caja(pos - 0.006, pos + 0.006, s0, s1, z0, z1)
        if eje == "x":
            col_caja(f"{nombre}_vidrio_{c:.1f}", s0, s1, pos - 0.01, pos + 0.01, z0, z1)
        else:
            col_caja(f"{nombre}_vidrio_{c:.1f}", pos - 0.01, pos + 0.01, s0, s1, z0, z1)


def puerta(c):
    return (c, PUERTA_W, 0.0, PUERTA_H, "puerta")


def ventana(c):
    return (c, VENT_W, VENT_Z0, VENT_Z1, "ventana")


def construir_paredes():
    ex0, ex1 = X0 - T / 2, X1 + T / 2
    D = lambda cara, s, z, w, h, ov, tipo, sala: dict(cara=cara, s=s, z=z, w=w, h=h, ov=ov, tipo=tipo, sala=sala)
    pared("ext_sur", "x", Y0, ex0, ex1, [ventana(-5), ventana(5)],
          danos=[D("Y", 3.2, 1.4, 1.4, 2.4, lambda W, Hh: ov_grieta(W, Hh, 37, 2), "grieta_estructural", "B")])
    pared("ext_norte", "x", Y1, ex0, ex1, [ventana(-5), ventana(5)],
          danos=[D("y", -3.2, 1.9, 2.0, 1.8, lambda W, Hh: ov_mancha(W, Hh, 3), "humedad", "C")]) 
    pared("ext_oeste", "y", X0, Y0 + T / 2, Y1 - T / 2, [ventana(0), ventana(6)])
    pared("ext_este", "y", X1, Y0 + T / 2, Y1 - T / 2, [ventana(0), ventana(6)])
    pared("int_x_m2", "y", -2.0, Y0, Y1, [puerta(0), puerta(6)], danos=[
        D("x", -1.7, 1.4, 1.8, 2.4, lambda W, Hh: ov_grieta(W, Hh, 11, 3), "grieta_estructural", "A"),
        D("X", 2.0, 1.5, 1.6, 2.4, lambda W, Hh: ov_grieta(W, Hh, 23, 4), "grieta_estructural", "vestibulo"),
        D("X", 4.6, 1.2, 1.3, 1.3, lambda W, Hh: ov_desprendimiento(W, Hh, 5), "desprendimiento_revoque", "vestibulo"),
    ])
    pared("int_x_p2", "y", 2.0, Y0, Y1, [puerta(0), (7.8, 1.3, 0.0, 1.7, "brecha")])
    pared("int_y_oeste", "x", 3.0, X0, -2.0, [puerta(-5)])
    pared("int_y_este", "x", 3.0, 2.0, X1, [puerta(5)])
    zocalos()


def zocalos():
    """Zocalo (rodapie) fino a lo largo de las paredes principales, de cada lado."""
    z = 0.10
    e = 0.012
    tramos = [
        # (eje, pos, a0, a1) sin descontar puertas: se interrumpen en los marcos y quedan escondidos por ellos
        ("x", Y0, X0, X1), ("x", Y1, X0, X1), ("y", X0, Y0, Y1), ("y", X1, Y0, Y1),
    ]
    for eje, pos, a0, a1 in tramos:
        for lado in (-1, 1):
            off = pos + lado * (T / 2 + e / 2)
            if eje == "x":
                M["marco"].caja(a0 + 0.1, a1 - 0.1, off - e / 2, off + e / 2, SLAB, z)
            else:
                M["marco"].caja(off - e / 2, off + e / 2, a0 + 0.1, a1 - 0.1, SLAB, z)


# ------------------------------------------------------------------ pisos y techo
def construir_pisos_techo():
    EXTRA_XML.append('<geom name="suelo_exterior" type="plane" pos="0 0 0.002" size="40 40 0.05" material="m_suelo" contype="0" conaffinity="0"/>')
    solido("piso_m", "x", X0, -2.0, Y0, 3.0, -0.05, SLAB, col=False)      # sala A
    solido("piso_m", "x", X0, -2.0, 3.0, Y1, -0.05, SLAB, col=False)      # sala C
    solido("piso_h", "x", -2.0, 2.0, Y0, Y1, -0.05, SLAB, col=False)      # vestibulo
    solido("piso_h", "x", 2.0, X1, Y0, 3.0, -0.05, SLAB, col=False)       # sala B
    solido("piso_h", "x", 2.0, X1, 3.0, Y1, -0.05, SLAB, col=False)       # sala D
    if TECHO:
        hx0, hx1, hy0, hy1 = 4.2, 5.9, -0.9, 0.7
        z0, z1 = H, H + 0.2
        ex0, ex1 = X0 - T / 2, X1 + T / 2
        ey0, ey1 = Y0 - T / 2, Y1 + T / 2
        t = M["techo"]
        t.caja(ex0, hx0, ey0, ey1, z0, z1)
        t.caja(hx1, ex1, ey0, ey1, z0, z1)
        t.caja(hx0, hx1, ey0, hy0, z0, z1)
        t.caja(hx0, hx1, hy1, ey1, z0, z1)
        # luminarias (paneles planos); la de la sala B (derrumbada) se omite
        for (cx, cy) in [(-5, 0), (-5, 6), (0, 1.5), (0, 6.5), (5, 6)]:
            M["luz"].caja(cx - 0.3, cx + 0.3, cy - 0.3, cy + 0.3, H - 0.025, H)
        MANIFEST.append(dict(tipo="techo_colapsado", sala="B", centro=[5.05, -0.1], z=H))
    else:
        MANIFEST.append(dict(tipo="escombros_de_techo", sala="B", centro=[5.05, -0.1]))


# ------------------------------------------------------------------ texturas de daño (overlays RGBA en [0,1])
def _ruido(w, h, escala, seed):
    r = np.random.default_rng(seed)
    base = r.random((max(2, h // escala), max(2, w // escala)))
    return np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)) / 255.0


def ov_grieta(w, h, seed=1, ramas=3):
    r = np.random.default_rng(seed)
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    esc = w / 1024.0

    def trazo(x, y, ang, largo, ancho, prof):
        paso = 7 * esc
        n = max(2, int(largo / paso))
        px, py = x, y
        for i in range(n):
            ang += r.normal(0, 0.3)
            x += paso * math.cos(ang)
            y += paso * math.sin(ang)
            a = max(1.0, ancho * (1 - i / n))
            d.line([(px, py), (x, y)], fill=255, width=int(round(a)))
            px, py = x, y
            if prof > 0 and r.random() < 0.02:
                trazo(x, y, ang + r.choice([-1, 1]) * r.uniform(0.5, 1.2), largo * 0.3, ancho * 0.6, prof - 1)

    trazo(w * 0.5 + r.uniform(-0.08, 0.08) * w, 0, math.pi / 2 + r.uniform(-0.3, 0.3), h * 1.05, 5 * esc, ramas)
    nucleo = np.array(m) / 255.0
    halo = gaussian_filter(nucleo, 3 * esc) * 0.35
    alpha = np.clip(np.maximum(nucleo, halo), 0, 1)
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:] = (42, 38, 34)
    return rgb, alpha


def ov_mancha(w, h, seed=3):
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w * 0.5, h * 0.55
    d = np.sqrt(((xx - cx) / (w * 0.45)) ** 2 + ((yy - cy) / (h * 0.45)) ** 2)
    ruido = _ruido(w, h, 40, seed) * 0.6 + _ruido(w, h, 12, seed + 1) * 0.4
    a = np.clip(1.1 - d + (ruido - 0.5) * 0.9, 0, 1)
    a = gaussian_filter(a, 3) ** 1.2
    borde = np.clip(a * 6, 0, 1) - np.clip(a * 6 - 0.5, 0, 1)
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[:] = np.array([66, 56, 40], np.float32) * (0.75 + 0.5 * ruido[..., None])
    rgb += borde[..., None] * 20
    return rgb, np.clip(a * 0.85, 0, 1)


def ov_desprendimiento(w, h, seed=5):
    """Revoque caido: deja ver el ladrillo debajo."""
    r = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * math.pi, 64, endpoint=False)
    rad = 0.36 + 0.10 * np.sin(ang * 3 + r.uniform(0, 6)) + 0.06 * r.normal(size=64)
    rad = gaussian_filter(rad, 1.0, mode="wrap")
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(
        [(w / 2 + w * rr * math.cos(a), h / 2 + h * rr * math.sin(a)) for a, rr in zip(ang, rad)], fill=255
    )
    m = np.array(m.filter(ImageFilter.GaussianBlur(1.2))) / 255.0
    ladr = Image.open(os.path.join(ASSETS, "ladrillo.png")).convert("RGB")
    ox, oy = int(r.integers(0, 1000)), int(r.integers(0, 1000))
    ladr = np.array(ladr.crop((ox, oy, ox + 1024, oy + 1024)).resize((w, h)), np.float32)
    prof = np.clip(gaussian_filter(m, 8) * 1.7, 0, 1)
    ladr *= 0.4 + 0.6 * prof[..., None]
    return ladr, m


_BASES = {}


def _base(nombre):
    if nombre not in _BASES:
        _BASES[nombre] = np.array(Image.open(os.path.join(ASSETS, nombre)).convert("RGB"), np.float32)
    return _BASES[nombre]


def hornear_decal(nombre, base_png, tile, centro, n, w, h, overlay, tipo, sala, extra=None):
    """Plano fino pegado a una pared. La textura es la de la pared (mismas coordenadas de mundo) + daño."""
    n = np.asarray(n, float)
    up = np.array([0, 0, 1.0])
    r = np.cross(up, n)  # con r x up == n
    if np.dot(np.cross(r, up), n) < 0:
        r = -r
    c = np.asarray(centro, float) + n * 0.002
    o = c - r * w / 2 - up * h / 2

    W, Hh = int(w * PX_POR_M), int(h * PX_POR_M)
    rgb_ov, a_ov = overlay(W, Hh)
    # coordenadas de mundo de cada pixel (fila 0 = v=0 = abajo del decal)
    uu = (np.arange(W) + 0.5) / W
    vv = (np.arange(Hh) + 0.5) / Hh
    P = o[None, None, :] + uu[None, :, None] * w * r[None, None, :] + vv[:, None, None] * h * up[None, None, :]
    base = _base(base_png)
    S = base.shape[0]
    along = P[..., 1] if abs(n[0]) > 0.5 else P[..., 0]
    col = ((along / tile) % 1.0 * S).astype(int) % S
    row = ((1.0 - (P[..., 2] / tile) % 1.0) * S).astype(int) % S
    fondo = base[row, col]
    out = fondo * (1 - a_ov[..., None]) + rgb_ov * a_ov[..., None]
    tex_nombre = f"decal_{nombre}.png"
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(ASSETS, tex_nombre))

    espesor = 0.003
    frente = [o, o + r * w, o + r * w + up * h, o + up * h]
    atras = [q - n * espesor for q in frente]
    m = Malla()
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    m._quad(frente, n, uv)
    m._quad(atras[::-1], -n, uv)
    for i in range(4):
        j = (i + 1) % 4
        nn = np.cross(frente[j] - frente[i], n)
        nn = nn / (np.linalg.norm(nn) + 1e-12)
        m._quad([frente[i], frente[j], atras[j], atras[i]], nn, [(0, 0)] * 4)
    m.guardar(nombre)
    DECALS.append((nombre, tex_nombre))
    MANIFEST.append(dict(tipo=tipo, sala=sala, centro=[float(x) for x in centro], normal=[float(x) for x in n],
                         ancho=w, alto=h, **(extra or {})))


DECALS = []


# ------------------------------------------------------------------ escombros (fisica real)
def crear_rocas(cuantas=10):
    nombres = []
    for i in range(cuantas):
        tam = RNG.uniform(0.09, 0.34)
        pts = RNG.normal(size=(16, 3))
        pts /= np.linalg.norm(pts, axis=1)[:, None]
        pts *= tam * RNG.uniform(0.6, 1.0, size=(16, 1)) * np.array([1.0, RNG.uniform(0.6, 1.0), RNG.uniform(0.4, 0.8)])
        m = Malla()
        m.poligono_convexo(pts, tile=0.5)
        nombres.append((m.guardar(f"roca_{i}"), pts))
    return nombres


def asentar_escombros(rocas, grupos):
    """grupos: lista de (cx, cy, zmin, zmax, radio, n). Devuelve poses finales tras simular caida."""
    comp = f'<compiler meshdir="{MESHES.replace(os.sep, "/")}"/>'
    assets = "".join(f'<mesh name="{n}" file="{n}.obj"/>' for n, _ in rocas)
    estatico = "".join(
        f'<geom type="box" pos="{cx} {cy} {cz}" size="{sx} {sy} {sz}"/>' for _, cx, cy, cz, sx, sy, sz in COL
    )
    cuerpos, k = [], 0
    for cx, cy, z0, z1, rad, n in grupos:
        for _ in range(n):
            nombre, _p = rocas[int(RNG.integers(0, len(rocas)))]
            ang, rr = RNG.uniform(0, 2 * math.pi), rad * math.sqrt(RNG.uniform())
            q = RNG.normal(size=4)
            q /= np.linalg.norm(q)
            cuerpos.append(
                f'<body name="r{k}" pos="{cx + rr*math.cos(ang):.3f} {cy + rr*math.sin(ang):.3f} {RNG.uniform(z0, z1):.3f}" '
                f'quat="{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {q[3]:.4f}"><freejoint/>'
                f'<geom type="mesh" mesh="{nombre}" density="2200" friction="0.9 0.02 0.001"/></body>'
            )
            k += 1
    xml = f"""<mujoco>{comp}<option timestep="0.002"/><asset>{assets}</asset><worldbody>
    <geom type="plane" size="0 0 0.05"/>{estatico}{''.join(cuerpos)}</worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(4000):
        mujoco.mj_step(model, data)
    poses = []
    for i in range(k):
        b = model.body(f"r{i}").id
        gm = model.body_geomadr[b]
        mesh_nombre = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[gm])
        poses.append((mesh_nombre, data.xpos[b].copy(), data.xquat[b].copy()))
    return poses


# ------------------------------------------------------------------ mobiliario
def mesa(cx, cy, lx=1.2, ly=0.7, h=0.74):
    solido("madera", f"mesa_{cx}_{cy}_tapa", cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2, h - 0.04, h)
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx * (lx / 2 - 0.05), cy + sy * (ly / 2 - 0.05)
            solido("madera", f"mesa_{cx}_{cy}_pata{sx}{sy}", px - 0.025, px + 0.025, py - 0.025, py + 0.025, 0, h - 0.04)


def silla(cx, cy, dir_y=1, h=0.45):
    solido("madera", f"silla_{cx}_{cy}_asiento", cx - 0.22, cx + 0.22, cy - 0.22, cy + 0.22, h - 0.04, h)
    for sx in (-1, 1):
        for sy in (-1, 1):
            px, py = cx + sx * 0.19, cy + sy * 0.19
            solido("madera", f"silla_{cx}_{cy}_pata{sx}{sy}", px - 0.02, px + 0.02, py - 0.02, py + 0.02, 0, h - 0.04)
    yb = cy - dir_y * 0.2
    solido("madera", f"silla_{cx}_{cy}_resp", cx - 0.22, cx + 0.22, yb - 0.02, yb + 0.02, h, h + 0.45)


def estante(cx, cy, lx=0.4, ly=1.2, h=1.9):
    # estante: laterales + baldas (la colision es el volumen completo)
    x0, x1, y0, y1 = cx - lx / 2, cx + lx / 2, cy - ly / 2, cy + ly / 2
    M["madera"].caja(x0, x1, y0, y0 + 0.03, 0, h)
    M["madera"].caja(x0, x1, y1 - 0.03, y1, 0, h)
    for zz in np.linspace(0.02, h - 0.03, 6):
        M["madera"].caja(x0, x1, y0, y1, zz, zz + 0.03)
    col_caja(f"estante_{cx}_{cy}", x0, x1, y0, y1, 0, h)


def escritorio(cx, cy):
    mesa(cx, cy, 1.4, 0.7, 0.75)
    silla(cx, cy - 0.7, dir_y=1)


SIN_COL = 'contype="0" conaffinity="0"'


def hoja_puerta(nombre, hx, hy, phi_deg, largo=0.92, alto=2.05):
    phi = math.radians(phi_deg)
    cx, cy = hx + math.cos(phi) * largo / 2, hy + math.sin(phi) * largo / 2
    EXTRA_XML.append(
        f'<geom name="{nombre}" type="box" pos="{cx:.3f} {cy:.3f} {alto/2:.3f}" size="{largo/2:.3f} 0.02 {alto/2:.3f}" '
        f'quat="{math.cos(phi/2):.4f} 0 0 {math.sin(phi/2):.4f}" material="m_madera" contype="0" conaffinity="0"/>'
    )


def caja_carton(nombre, cx, cy, lx=0.5, ly=0.4, lz=0.35, yaw=0.0, apilada=False):
    a = math.radians(yaw)
    EXTRA_XML.append(
        f'<geom name="{nombre}" type="box" pos="{cx} {cy} {lz/2 + (0.35 if apilada else 0)}" size="{lx/2} {ly/2} {lz/2}" '
        f'quat="{math.cos(a/2):.4f} 0 0 {math.sin(a/2):.4f}" rgba="0.40 0.31 0.21 1" '
        f'{SIN_COL if apilada else ""}/>'
    )


def utileria():
    # puertas entre salas y vestibulo, abiertas en distintos angulos (el hueco queda libre)
    hoja_puerta("puerta_A_vest", -2.0, -0.5, 15)        # sala A <-> vestibulo, abierta hacia el vestibulo
    hoja_puerta("puerta_B_vest", 2.0, 0.5, 195)          # sala B, abierta hacia el vestibulo
    hoja_puerta("puerta_C_vest", -2.0, 5.5, 20)
    hoja_puerta("puerta_A_C", -5.5, 3.0, 75)
    hoja_puerta("puerta_B_D", 5.5, 3.0, 105)
    # cajas
    caja_carton("carton_1", 7.4, 7.9, yaw=20)
    caja_carton("carton_2", 7.35, 7.35, 0.45, 0.4, 0.3, yaw=-15)
    caja_carton("carton_3", 7.4, 7.85, 0.4, 0.35, 0.3, yaw=5, apilada=True)
    caja_carton("carton_4", -7.2, 4.0, yaw=35)


def mobiliario():
    utileria()
    mesa(-5.2, 0.8, 1.8, 0.9)
    silla(-5.7, 1.6, dir_y=-1)
    silla(-4.7, 1.6, dir_y=-1)
    silla(-5.2, -0.1, dir_y=1)
    estante(-7.75, -1.8)
    escritorio(-6.0, 6.0)
    escritorio(-4.0, 7.8)
    estante(-7.75, 8.0, ly=1.4)
    escritorio(5.0, 5.0)
    escritorio(6.8, 7.5)
    mesa(6.9, -1.8, 1.2, 0.7)


# ------------------------------------------------------------------ daño
def construir_dano():
    # Sala C: desprendimiento extra en la pared norte
    # (los daños de superficie se hornean dentro de las piezas de pared: ver construir_paredes)
    # Viga caida en la sala B
    ang = math.radians(18)
    EXTRA_XML.append(
        f'<geom name="viga_caida" type="box" pos="4.9 0.0 0.55" size="1.4 0.14 0.14" quat="{math.cos(ang/2):.4f} 0 {math.sin(ang/2):.4f} 0" material="m_hormigon"/>'
    )
    MANIFEST.append(dict(tipo="viga_caida", sala="B", centro=[4.9, 0.0]))

    # Paneles de cielorraso colgando
    for i, (px, py, pz, roll) in enumerate(([(4.2, 0.0, 2.55, 38), (5.9, 0.4, 2.45, -52)] if TECHO else [(4.4, 0.55, 0.45, 20), (5.7, -0.6, 0.4, -15)])):
        a = math.radians(roll)
        EXTRA_XML.append(
            f'<geom name="panel_colgante_{i}" type="box" pos="{px} {py} {pz}" size="0.45 0.3 0.015" '
            f'quat="{math.cos(a/2):.4f} 0 {math.sin(a/2):.4f} 0" material="m_revoque" contype="0" conaffinity="0"/>'
        )

    # Brecha en la pared x=2 (sala D <-> vestibulo): bordes irregulares sobre la abertura
    jag = [(7.25, 1.85, 0.18), (7.45, 1.95, 0.22), (7.7, 1.8, 0.15), (7.95, 2.1, 0.25), (8.2, 1.85, 0.18), (8.35, 1.82, 0.1)]
    for i, (yc, zt, ancho) in enumerate(jag):
        solido("revoque", f"brecha_borde_{i}", 2.0 - T / 2, 2.0 + T / 2, yc - ancho / 2, yc + ancho / 2, 1.7, zt)
    MANIFEST.append(dict(tipo="brecha_pared", sala="D/vestibulo", centro=[2.0, 7.8], ancho=1.3, alto=1.7))

    rocas = crear_rocas(10)
    grupos = [
        (5.05, -0.1, 0.4, 2.0, 0.95, 48),
        (1.5, 7.8, 0.6, 2.0, 0.5, 20),
        (2.7, 7.8, 0.6, 2.0, 0.55, 22),
        (-4.0, 6.0, 0.3, 0.8, 0.6, 5),
    ]
    poses = asentar_escombros(rocas, grupos)
    return rocas, poses


# ------------------------------------------------------------------ camaras
def look_at(nombre, pos, target, fovy=70):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    z = np.array([0, 0, 1.0])
    r = np.cross(f, z)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return (f'<camera name="{nombre}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
            f'xyaxes="{r[0]:.4f} {r[1]:.4f} {r[2]:.4f} {u[0]:.4f} {u[1]:.4f} {u[2]:.4f}" fovy="{fovy}"/>')


CAMARAS = [
    ("inf_vista_aerea", (0, -17, 13), (0, 3, 0.8), 45),
    ("inf_planta", (0, 3, 24), (0, 3, 0), 45),
    ("inf_exterior", (0, -11, 1.7), (0, 0, 1.4), 60),
    ("inf_vestibulo", (0, -2.55, 1.55), (0, 6, 1.2), 70),
    ("inf_sala_a", (-7.55, -2.55, 1.5), (-4, 1.2, 1.1), 75),
    ("inf_sala_a_grieta", (-5.6, -0.4, 1.5), (-2.1, -1.9, 1.35), 60),
    ("inf_sala_b_techo", (7.5, -2.6, 1.4), (4.9, 0.2, 1.7), 75),
    ("inf_brecha", (-0.7, 3.3, 1.5), (2.0, 7.8, 0.9), 70),
    ("inf_sala_c_humedad", (-6.6, 4.9, 1.5), (-3.2, 8.9, 1.7), 70),
]


# ------------------------------------------------------------------ XML final
def escribir_xml(rocas, poses):
    assets, geoms = [], []

    def malla_mat(clave, material, grupo=0):
        if not M[clave].f:
            return
        n = M[clave].guardar(f"m_{clave}")
        assets.append(f'<mesh name="{n}" file="{PREF}/meshes/{n}.obj"/>')
        geoms.append(f'<geom name="vis_{clave}" type="mesh" mesh="{n}" material="{material}" group="{grupo}" contype="0" conaffinity="0"/>')

    malla_mat("revoque", "m_revoque")
    malla_mat("hormigon", "m_hormigon")
    malla_mat("piso_h", "m_piso_h")
    malla_mat("piso_m", "m_piso_m")
    malla_mat("suelo", "m_suelo")
    malla_mat("techo", "m_techo", grupo=1)
    malla_mat("madera", "m_madera")
    malla_mat("marco", "m_marco")
    malla_mat("luz", "m_luz", grupo=1)
    malla_mat("vidrio", "m_vidrio", grupo=1)

    for nombre, tex in PIEZAS_DANADAS:
        assets.append(f'<texture name="t_{nombre}" type="2d" file="{PREF}/assets/{tex}"/>')
        assets.append(f'<material name="mt_{nombre}" texture="t_{nombre}" rgba="1 1 1 1" specular="0.05" shininess="0.05"/>')
        assets.append(f'<mesh name="{nombre}" file="{PREF}/meshes/{nombre}.obj"/>')
        geoms.append(f'<geom name="vis_{nombre}" type="mesh" mesh="{nombre}" material="mt_{nombre}" group="0" contype="0" conaffinity="0"/>')
    for nombre, tex in DECALS:
        assets.append(f'<texture name="t_{nombre}" type="2d" file="{PREF}/assets/{tex}"/>')
        assets.append(f'<material name="mt_{nombre}" texture="t_{nombre}" rgba="1 1 1 1" specular="0.05" shininess="0.05"/>')
        assets.append(f'<mesh name="{nombre}" file="{PREF}/meshes/{nombre}.obj"/>')
        geoms.append(f'<geom name="vis_{nombre}" type="mesh" mesh="{nombre}" material="mt_{nombre}" group="0" contype="0" conaffinity="0"/>')

    for n, _ in rocas:
        assets.append(f'<mesh name="{n}" file="{PREF}/meshes/{n}.obj"/>')
    rocas_xml = []
    for i, (n, p, q) in enumerate(poses):
        rocas_xml.append(
            f'<geom name="escombro_{i}" type="mesh" mesh="{n}" material="m_escombro" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" friction="0.9 0.02 0.001"/>'
        )

    cols = [
        f'<geom name="col_{n}" type="box" pos="{cx:.3f} {cy:.3f} {cz:.3f}" size="{sx:.3f} {sy:.3f} {sz:.3f}" group="3" rgba="0.9 0.2 0.2 0.3"/>'
        for n, cx, cy, cz, sx, sy, sz in COL
    ]

    luces = []
    for i, (lx, ly) in enumerate([(-5, 0), (-5, 6), (0, 1.5), (0, 6.5), (5, 6), (5, 0)]):
        dim = 0.45 if (lx, ly) == (5, 0) else 0.85
        luces.append(f'<light name="luz_{i}" pos="{lx} {ly} 2.65" dir="0 0 -1" cutoff="110" exponent="1" diffuse="{dim} {dim} {dim*0.95:.3f}" specular="0.1 0.1 0.1" attenuation="1 0.03 0.01" castshadow="false"/>')
    luces.append('<light name="sol" directional="true" pos="-20 0 20" dir="0.65 0.12 -0.55" diffuse="1.5 1.3 1.05" specular="0.2 0.2 0.2" castshadow="true"/>')
    luces.append('<light name="relleno" directional="true" pos="0 0 10" dir="-0.2 0.1 -1" diffuse="0.12 0.13 0.16" castshadow="false"/>')

    cams = [look_at(*c) for c in CAMARAS]

    xml = f"""<mujocoinclude>
  <!-- GENERADO por generar_edificio.py, no editar a mano. -->
  <statistic extent="20" center="0 3 1"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="8192"/>
    <headlight ambient="0.26 0.26 0.28" diffuse="0.12 0.12 0.12" specular="0.05 0.05 0.05"/>
    <map znear="0.002" zfar="120" shadowscale="0.6"/>
  </visual>
  <asset>
    <texture name="t_revoque" type="2d" file="{PREF}/assets/pared_revoque.png"/>
    <texture name="t_hormigon" type="2d" file="{PREF}/assets/pared_hormigon.png"/>
    <texture name="t_piso_h" type="2d" file="{PREF}/assets/piso_hormigon.png"/>
    <texture name="t_piso_m" type="2d" file="{PREF}/assets/piso_madera.png"/>
    <texture name="t_mueble" type="2d" file="{PREF}/assets/mueble_madera.png"/>
    <texture name="t_escombro" type="2d" file="{PREF}/assets/escombro.png"/>
    <material name="m_revoque" texture="t_revoque" rgba="1 1 1 1" specular="0.05" shininess="0.05"/>
    <material name="m_hormigon" texture="t_hormigon" rgba="0.95 0.95 0.95 1" specular="0.05" shininess="0.05"/>
    <material name="m_piso_h" texture="t_piso_h" rgba="1 1 1 1" specular="0.15" shininess="0.2" reflectance="0.05"/>
    <material name="m_piso_m" texture="t_piso_m" rgba="1 1 1 1" specular="0.2" shininess="0.3" reflectance="0.05"/>
    <material name="m_suelo" texture="t_piso_h" texuniform="true" texrepeat="0.35 0.35" rgba="0.8 0.8 0.8 1" specular="0.05" shininess="0.05"/>
    <material name="m_techo" texture="t_revoque" rgba="1 1 1 1" specular="0.02" shininess="0.02" emission="0.35"/>
    <material name="m_madera" texture="t_mueble" rgba="1 1 1 1" specular="0.2" shininess="0.3"/>
    <material name="m_marco" rgba="0.20 0.13 0.08 1" specular="0.2" shininess="0.3"/>
    <material name="m_luz" rgba="1 1 0.92 1" emission="1"/>
    <material name="m_vidrio" rgba="0.75 0.88 0.95 0.15" specular="0.6" shininess="0.8"/>
    <material name="m_escombro" texture="t_escombro" rgba="1 1 1 1" specular="0.05" shininess="0.05"/>
    {chr(10).join('    ' + a for a in assets)}
  </asset>
  <worldbody>
    {chr(10).join('    ' + l for l in luces)}
    <body name="edificio_informe">
      {chr(10).join('      ' + g for g in geoms)}
      {chr(10).join('      ' + g for g in cols)}
      {chr(10).join('      ' + g for g in EXTRA_XML)}
      {chr(10).join('      ' + g for g in rocas_xml)}
    </body>
    {chr(10).join('    ' + c for c in cams)}
  </worldbody>
</mujocoinclude>
"""
    with open(os.path.join(AQUI, "escenario.xml"), "w", encoding="utf-8") as fh:
        fh.write(xml)
    with open(os.path.join(AQUI, "damage_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(MANIFEST, fh, indent=2, ensure_ascii=False)


def main():
    construir_paredes()
    construir_pisos_techo()
    mobiliario()
    rocas, poses = construir_dano()
    escribir_xml(rocas, poses)
    print(f"ok: {len(COL)} cajas de colision, {len(poses)} escombros, {len(DECALS)} decals")


if __name__ == "__main__":
    main()





