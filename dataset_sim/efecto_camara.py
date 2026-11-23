"""
Aplica a capturas del simulador los defectos de una camara real: exposicion y
balance de blancos desparejos, curva de tono, viñeteo, un poco de desenfoque,
ruido del sensor y compresion JPEG. Las fotos del simulador salen perfectas y
las de la camara del robot no.

No toca las capturas originales: escribe una copia con la misma estructura de
carpetas y el mismo capturas.csv.

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/efecto_camara.py --entrada dataset_sim/capturas_procthor_real \
        --salida dataset_sim/capturas_procthor_real_cam
"""
import argparse
import glob
import io
import os
import shutil

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def efecto(img, r):
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    H, W = a.shape[:2]
    a = a * r.uniform(0.8, 1.2) * r.uniform(0.94, 1.06, 3)
    a = np.clip(a, 0, 1) ** r.uniform(0.85, 1.15)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    a = a * (1 - r.uniform(0.05, 0.3) * d2 / 2)[..., None]
    a = gaussian_filter(a, (r.uniform(0.3, 1.1),) * 2 + (0,))
    ruido = r.uniform(0.006, 0.025)
    a = a + r.normal(0, 1, a.shape) * ruido * np.sqrt(np.clip(a, 0.02, 1))
    a = a + gaussian_filter(r.normal(0, ruido * 0.6, a.shape), (1.5, 1.5, 0))
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=int(r.integers(55, 92)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()
    r = np.random.default_rng(args.semilla)
    fotos = sorted(glob.glob(os.path.join(args.entrada, "*", "*", "*.png")))
    for ruta in fotos:
        destino = os.path.join(args.salida, os.path.relpath(ruta, args.entrada))
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        efecto(Image.open(ruta), r).save(destino)
    shutil.copy(os.path.join(args.entrada, "capturas.csv"), os.path.join(args.salida, "capturas.csv"))
    print(f"listo: {len(fotos)} fotos en {args.salida}")


if __name__ == "__main__":
    main()
