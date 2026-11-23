"""
Genera edificios "realistas": todo lo que se ve sale de fotos reales.

  - Paredes, pisos y muebles: texturas fotograficas, sin pintar de un color plano.
  - Grietas: fotos reales de grietas (las mismas del conjunto de entrenamiento,
    carpeta data_superficies de humanoid-damage-cnn), pegadas sobre la pared
    con el color de la pared y los bordes difuminados.
  - Humedad y desprendimientos: recortes de fotos reales de revoque deteriorado.
  - Con techo y luces de sala, como un interior real.

Usa el mismo generador que generar_variantes.py. Salida:
    escenarios/variantes/edificio_real_NN/

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/generar_realista.py --desde 1 --hasta 3
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

import generar_variantes as gv

GRIETAS = os.path.join(os.path.dirname(gv.RAIZ), "humanoid-damage-cnn", "notebooks", "data_superficies",
                       "5y9wdsg2zt-2", "Concrete Crack Images for Classification", "Positive")
DETERIORO = sorted(glob.glob(os.path.join(gv.TEXTURAS, "dano__*.jpg")))
_fotos_grietas = None


def borde_difuso(W, Hh, rng, irregular=True):
    """Mascara con bordes difuminados para que el recorte se funda con la pared."""
    yy, xx = np.mgrid[0:Hh, 0:W]
    d = np.sqrt(((xx - W / 2) / (W * 0.5)) ** 2 + ((yy - Hh / 2) / (Hh * 0.5)) ** 2)
    if irregular:
        ruido = np.array(Image.fromarray((rng.random((8, 8)) * 255).astype(np.uint8)).resize((W, Hh), Image.BICUBIC)) / 255.0
        d = d + (ruido - 0.5) * 0.35
    return np.clip((0.92 - d) / 0.22, 0, 1)


def overlay_real(tipo, w, h, rng):
    global _fotos_grietas
    if _fotos_grietas is None:
        _fotos_grietas = sorted(glob.glob(os.path.join(GRIETAS, "*.jpg")))
    semilla = int(rng.integers(1, 10 ** 6))

    def ov(W, Hh):
        r = np.random.default_rng(semilla)
        W, Hh = max(16, W), max(16, Hh)
        pared = np.array(Image.open(os.path.join(_assets[0], "pared_revoque.png")).convert("RGB"), np.float32)
        color = pared.reshape(-1, 3).mean(axis=0)
        if tipo == "grieta_estructural":
            # mosaico de fotos reales de grietas: cada una cubre unos 45 cm de pared
            lado = max(32, int(0.45 * W / w))
            nx, ny = max(1, round(W / lado)), max(1, round(Hh / lado))
            lienzo = np.zeros((ny * lado, nx * lado, 3), np.float32)
            alfa = np.zeros((ny * lado, nx * lado), np.float32)
            for j in range(ny):
                for i in range(nx):
                    foto = Image.open(_fotos_grietas[r.integers(len(_fotos_grietas))]).convert("RGB")
                    foto = foto.rotate(int(r.integers(4)) * 90).resize((lado, lado), Image.LANCZOS)
                    a = np.asarray(foto, np.float32)
                    gris = a.mean(axis=2)
                    relieve = gris / np.maximum(gaussian_filter(gris, lado / 6), 1.0)
                    lienzo[j * lado:(j + 1) * lado, i * lado:(i + 1) * lado] = color[None, None, :] * relieve[..., None]
                    alfa[j * lado:(j + 1) * lado, i * lado:(i + 1) * lado] = borde_difuso(lado, lado, r, irregular=False)
            rgb = np.asarray(Image.fromarray(np.clip(lienzo, 0, 255).astype(np.uint8)).resize((W, Hh), Image.LANCZOS), np.float32)
            a = np.asarray(Image.fromarray((alfa * 255).astype(np.uint8)).resize((W, Hh), Image.BILINEAR), np.float32) / 255.0
            return rgb, a
        # humedad / desprendimiento: recorte de una foto real de revoque deteriorado
        foto = Image.open(DETERIORO[r.integers(len(DETERIORO))]).convert("RGB")
        lado = int(foto.size[0] * min(1.0, max(w, h) / 2.0))
        ox, oy = int(r.integers(0, foto.size[0] - lado + 1)), int(r.integers(0, foto.size[1] - lado + 1))
        rgb = np.asarray(foto.crop((ox, oy, ox + lado, oy + lado)).resize((W, Hh), Image.LANCZOS), np.float32)
        if tipo == "humedad":
            rgb = rgb * 0.75
        return rgb, borde_difuso(W, Hh, r)
    return ov


_assets = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1)
    ap.add_argument("--hasta", type=int, default=1)
    args = ap.parse_args()
    gv.PROB_PINTAR = 0.0
    gv.OVERLAY_REAL = overlay_real
    os.makedirs(gv.DESTINO, exist_ok=True)
    for n in range(args.desde, args.hasta + 1):
        nombre = f"edificio_real_{n:02d}"
        _assets[0] = os.path.join(gv.DESTINO, nombre, "assets")
        gv.generar(500 + n, nombre=nombre, techo=True)


if __name__ == "__main__":
    main()
