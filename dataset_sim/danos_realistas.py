"""
Daños de pared y texturas con aspecto mas real, para las casas de ProcTHOR.

  - Paredes: revoque liso pintado en colores claros, como un interior real.
    Solo queda un grano muy fino de la textura fotografica.
  - Grietas: trazo irregular con el interior oscuro, bordes astillados y
    pequeños saltos de pintura a lo largo.
  - Humedad: sube desde el piso o baja del techo. Tiene tres aspectos: mancha
    amarronada con marca de agua y pintura descascarada, moho negro verdoso, o
    salitre blanco con revoque flojo cerca del piso.
  - Desprendimiento: revoque caido que deja ver el ladrillo o el revoque
    grueso, con sombra en el borde y pintura levantada alrededor.

Uso: generar_procthor.py --estilo realista
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter

COLORES_PARED = [(236, 234, 228), (228, 224, 214), (240, 236, 224), (222, 226, 228), (232, 228, 236),
                 (226, 232, 222), (238, 228, 214), (214, 214, 210), (244, 242, 236), (230, 222, 206)]
ASSETS = [None]
LADRILLO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "escenarios", "edificio_informe",
                        "assets", "ladrillo.png")


def ruido(w, h, escala, r):
    base = r.random((max(2, int(h / escala)), max(2, int(w / escala))))
    return np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC), np.float32) / 255.0


def fbm(w, h, r, escalas=(160, 60, 22, 8), pesos=(0.5, 0.27, 0.15, 0.08)):
    return sum(p * ruido(w, h, e, r) for e, p in zip(escalas, pesos))


def pintar_pared(ruta_textura, destino, rng, lado=1024):
    """Revoque liso pintado: color claro, grano fino y una variacion suave de tono."""
    a = np.asarray(Image.open(ruta_textura).convert("RGB").resize((lado, lado), Image.LANCZOS), np.float32)
    g = a.mean(axis=2)
    grano = g / np.maximum(gaussian_filter(g, 6, mode="wrap"), 1.0) - 1.0
    color = np.array(COLORES_PARED[rng.integers(len(COLORES_PARED))], np.float32)
    suave = gaussian_filter(rng.random((lado, lado)), 60, mode="wrap")
    suave = (suave - suave.mean()) / max(1e-6, suave.std())
    out = color[None, None, :] * (1.0 + rng.uniform(0.03, 0.06) * grano + 0.003 * suave)[..., None]
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(destino)
    return [int(c) for c in color]


def color_pared():
    img = np.asarray(Image.open(os.path.join(ASSETS[0], "pared_revoque.png")).convert("RGB"), np.float32)
    return img.reshape(-1, 3).mean(axis=0)


def grieta(W, Hh, r, color):
    """Grieta estructural: trazo ancho con el interior oscuro y el revoque saltado a lo largo."""
    m = Image.new("L", (W, Hh), 0)
    d = ImageDraw.Draw(m)
    puntos = []

    def trazo(x, y, ang, largo, ancho, prof):
        paso = 3.0
        n = max(2, int(largo / paso))
        px, py = x, y
        rumbo = ang
        for i in range(n):
            ang += r.normal(0, 0.16) + 0.12 * (rumbo - ang) + (0.4 if r.random() < 0.03 else 0) * r.choice([-1, 1])
            x += paso * math.cos(ang)
            y += paso * math.sin(ang)
            a = max(1.5, ancho * (1 - 0.6 * i / n) * r.uniform(0.6, 1.4))
            d.line([(px, py), (x, y)], fill=255, width=int(round(a)))
            puntos.append((x, y, a))
            px, py = x, y
            if prof > 0 and r.random() < 0.006:
                trazo(x, y, ang + r.choice([-1, 1]) * r.uniform(0.4, 1.1), largo * r.uniform(0.2, 0.45),
                      ancho * 0.5, prof - 1)

    if r.random() < 0.5:
        inicio, ang = (W * r.uniform(0.25, 0.75), 0), math.pi / 2 + r.uniform(-0.6, 0.6)
    else:
        inicio, ang = (0, Hh * r.uniform(0.0, 0.3)), r.uniform(0.5, 1.0)
    ancho = r.uniform(6, 12)
    trazo(inicio[0], inicio[1], ang, math.hypot(W, Hh) * 1.0, ancho, 2)
    nucleo = np.asarray(m.filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255.0
    # revoque saltado: manchas irregulares a lo largo de la grieta donde se ve el material de abajo
    saltado = Image.new("L", (W, Hh), 0)
    ds = ImageDraw.Draw(saltado)
    for x, y, a in puntos:
        if a > 4 and r.random() < 0.025:
            rr = a * r.uniform(1.2, 2.5)
            ds.ellipse([x - rr, y - rr * r.uniform(0.5, 1.0), x + rr, y + rr * r.uniform(0.5, 1.0)], fill=255)
    rugoso = fbm(W, Hh, r, (10, 4, 2), (0.5, 0.3, 0.2))
    saltado = (gaussian_filter(np.asarray(saltado, np.float32) / 255.0, 2) + (rugoso - 0.5) * 0.6) > 0.45
    halo = gaussian_filter(nucleo, 3.0)
    sucio = gaussian_filter(nucleo, 9.0)
    revoque = np.array([168, 160, 148], np.float32)[None, None, :] * (0.75 + 0.5 * rugoso[..., None])
    rgb = np.empty((Hh, W, 3), np.float32)
    rgb[:] = color * 0.70
    rgb[saltado] = revoque[saltado]
    rgb = rgb * (1 - nucleo[..., None]) + np.array([30, 27, 24], np.float32) * nucleo[..., None]
    alfa = np.clip(np.maximum.reduce([nucleo, saltado * 0.95, halo * 0.8, sucio * 0.35]), 0, 1)
    return rgb, alfa


def _descascarado(rgb, alfa, zona, W, Hh, r, color, escala):
    """Pintura que se levanta en escamas: deja ver el revoque gris, con borde claro y sombra."""
    from scipy.ndimage import binary_erosion, shift
    escamas = (fbm(W, Hh, r, (escala, escala / 3, 4), (0.55, 0.3, 0.15)) > 0.56) & (zona > 0.35)
    revoque = np.array([176, 168, 156], np.float32)[None, None, :] * (0.8 + 0.4 * fbm(W, Hh, r, (8, 3), (0.6, 0.4)))[..., None]
    rgb[escamas] = revoque[escamas]
    borde = escamas & ~binary_erosion(escamas, iterations=2)
    sombra = shift(escamas.astype(np.float32), (2, 2), order=0) > 0.5
    sombra &= ~escamas
    rgb[borde] = rgb[borde] * 0.35 + np.array([245, 243, 238], np.float32) * 0.65
    rgb[sombra] = rgb[sombra] * 0.55
    alfa[escamas | borde | sombra] = 1.0
    return rgb, alfa


def humedad(W, Hh, r, color, desde_abajo):
    """Humedad con tres aspectos comunes:
       - mancha amarronada con marca de agua y pintura descascarada
       - moho negro verdoso en manchas densas, mas fuerte hacia el origen del agua
       - salitre: polvo blanco y revoque flojo cerca del piso
    """
    yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
    v = (yy / Hh) if not desde_abajo else (1 - yy / Hh)
    lateral = np.abs(xx / W - 0.5) * 2
    n = gaussian_filter(fbm(W, Hh, r, (120, 50, 20), (0.5, 0.3, 0.2)), 3)
    campo = 1.2 - v * 1.3 - lateral ** 2.5 * 0.6 + (n - 0.5) * 0.8
    a = gaussian_filter(np.clip(campo * 2.2, 0, 1), 4)
    intensidad = np.clip(campo, 0, 1)
    estilo = r.choice(["mancha", "moho", "salitre"] if desde_abajo else ["mancha", "moho"],
                      p=[0.4, 0.3, 0.3] if desde_abajo else [0.55, 0.45])
    tono = 0.85 + 0.3 * gaussian_filter(fbm(W, Hh, r, (40, 12), (0.6, 0.4)), 2)
    if estilo == "mancha":
        borde = np.exp(-((campo - 0.25) / 0.06) ** 2)
        marron = np.array([126, 100, 64], np.float32)
        mezcla = (0.35 + 0.5 * intensidad)[..., None]
        rgb = (color[None, None, :] * (1 - mezcla) + marron[None, None, :] * mezcla) * tono[..., None]
        rgb = rgb * (1 - 0.45 * borde[..., None]) + np.array([86, 62, 36], np.float32) * 0.45 * borde[..., None]
        alfa = np.clip(np.maximum(a * 0.95, borde * 0.9), 0, 1)
        if r.random() < 0.8:
            rgb, alfa = _descascarado(rgb, alfa, intensidad, W, Hh, r, color, r.uniform(25, 60))
    elif estilo == "moho":
        gris = np.array([150, 146, 132], np.float32)
        mezcla = (0.2 + 0.45 * intensidad)[..., None]
        rgb = (color[None, None, :] * (1 - mezcla) + gris[None, None, :] * mezcla) * tono[..., None]
        densidad = intensidad ** 1.5
        puntos = fbm(W, Hh, r, (14, 5, 2), (0.45, 0.35, 0.2))
        moho = puntos > (0.78 - 0.32 * densidad)
        oscuro = np.array([30, 33, 26], np.float32)
        rgb[moho] = rgb[moho] * 0.2 + oscuro * 0.8
        halo = gaussian_filter(moho.astype(np.float32), 3)
        rgb = rgb * (1 - 0.35 * halo[..., None]) + oscuro * 0.35 * halo[..., None]
        alfa = np.clip(np.maximum(a * 0.85, halo), 0, 1)
        alfa[moho] = 1.0
    else:
        blanco = np.array([238, 236, 230], np.float32)
        rgb = color[None, None, :] * (0.9 + 0.1 * tono[..., None])
        humedo = np.clip(intensidad * 1.2, 0, 1)[..., None]
        rgb = rgb * (1 - 0.25 * humedo) + np.array([120, 104, 80], np.float32) * 0.25 * humedo
        polvo = gaussian_filter((fbm(W, Hh, r, (20, 6, 2), (0.5, 0.3, 0.2)) > 0.5).astype(np.float32), 1.5) * intensidad
        rgb = rgb * (1 - 0.8 * polvo[..., None]) + blanco * 0.8 * polvo[..., None]
        alfa = np.clip(np.maximum(a * 0.9, polvo), 0, 1)
        rgb, alfa = _descascarado(rgb, alfa, intensidad * 1.2, W, Hh, r, color, r.uniform(15, 35))
    return rgb, alfa


def desprendimiento(W, Hh, r, color):
    """Revoque caido: zona irregular con bordes rotos, ladrillo o revoque grueso a la vista y escombro."""
    yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
    cx, cy = W * r.uniform(0.4, 0.6), Hh * r.uniform(0.4, 0.6)
    dist = np.sqrt(((xx - cx) / (W * 0.42)) ** 2 + ((yy - cy) / (Hh * 0.42)) ** 2)
    forma = dist + (fbm(W, Hh, r, (90, 35, 12, 5), (0.4, 0.3, 0.2, 0.1)) - 0.5) * 0.9
    hueco = np.clip((1.0 - forma) * 12, 0, 1)
    anillo = np.clip((1.18 - forma) * 12, 0, 1) * (1 - hueco)
    if r.random() < 0.6:
        fondo = Image.open(LADRILLO).convert("RGB")
        lado = int(r.integers(450, 800))
        ox, oy = int(r.integers(0, fondo.size[0] - lado)), int(r.integers(0, fondo.size[1] - lado))
        fondo = np.asarray(fondo.crop((ox, oy, ox + lado, oy + lado)).resize((W, Hh)), np.float32)
        fondo = fondo * 0.85 + np.array([150, 140, 128], np.float32) * 0.15
    else:
        g = fbm(W, Hh, r, (30, 10, 4, 2), (0.3, 0.3, 0.2, 0.2))
        fondo = np.array([150, 144, 134], np.float32)[None, None, :] * (0.65 + 0.7 * g[..., None])
    restos = (fbm(W, Hh, r, (14, 5), (0.6, 0.4)) > 0.62) & (hueco > 0.5)
    fondo[restos] = fondo[restos] * 0.3 + np.array([176, 168, 156], np.float32) * 0.7
    sombra = np.clip(hueco - gaussian_filter(hueco, 5), 0, 1) * 2.5
    fondo = fondo * (1 - 0.6 * np.clip(sombra, 0, 1)[..., None])
    canto = np.array([196, 190, 180], np.float32)
    rgb = fondo * hueco[..., None] + (canto * 0.5 + color * 0.4)[None, None, :] * anillo[..., None]
    rgb += (color * 0.72)[None, None, :] * (1 - hueco - anillo)[..., None]
    sucio = gaussian_filter(hueco, 14)
    alfa = np.clip(hueco + anillo + sucio * 0.4, 0, 1)
    return rgb, alfa


def overlay(tipo, w, h, rng, desde_abajo=True):
    semilla = int(rng.integers(1, 10 ** 6))

    def ov(W, Hh):
        r = np.random.default_rng(semilla)
        W, Hh = max(16, W), max(16, Hh)
        color = color_pared()
        if tipo == "grieta_estructural":
            return grieta(W, Hh, r, color)
        if tipo == "humedad":
            return humedad(W, Hh, r, color, desde_abajo)
        return desprendimiento(W, Hh, r, color)
    return ov


def textura_tela(destino, rng, lado=256):
    """Tela de tapizado: trama fina gris clara para multiplicar por el color del mueble."""
    yy, xx = np.mgrid[0:lado, 0:lado]
    trama = 0.5 + 0.25 * np.sin(xx * 2.2) * np.sin(yy * 2.2)
    g = 0.86 + 0.08 * trama + 0.06 * gaussian_filter(rng.random((lado, lado)), 1.2, mode="wrap")
    Image.fromarray(np.clip(g * 255, 0, 255).astype(np.uint8)).convert("RGB").save(destino)
