"""Descarga texturas CC0 de Poly Haven (diffuse 2k) y las guarda como PNG
en escenarios/edificio_informe/assets/. Uso: python descargar_texturas.py"""
import io
import os
import requests
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(AQUI, "assets")
os.makedirs(DEST, exist_ok=True)

# nombre_local -> id de Poly Haven
TEXTURAS = {
    "pared_hormigon": "concrete_wall_008",
    "pared_revoque": "white_plaster_02",
    "piso_hormigon": "concrete_floor_worn_001",
    "piso_madera": "old_wood_floor",
    "ladrillo": "brick_wall_005",
    "mueble_madera": "dark_wood",
    "escombro": "concrete_debris",
}
RES = "2k"


def pintura_blanca(img, blanco=(196, 196, 194), fuerza=0.30, sigma=40):
    """Pared pintada de blanco liso: conserva solo el grano fino de la textura
    (se quitan manchones de baja frecuencia y el color) sobre un blanco plano.
    El filtro usa wrap, asi que el mosaico sigue siendo continuo."""
    import numpy as np
    from scipy.ndimage import gaussian_filter
    a = np.asarray(img).astype(np.float32)
    g = a.mean(axis=2)
    bajo = gaussian_filter(g, sigma=sigma, mode="wrap")
    hp = g / np.maximum(bajo, 1.0) - 1.0
    out = np.array(blanco, np.float32)[None, None, :] * (1.0 + fuerza * hp)[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

for local, ph_id in TEXTURAS.items():
    salida = os.path.join(DEST, f"{local}.png")
    if os.path.exists(salida):
        print("ya existe", local)
        continue
    info = requests.get(f"https://api.polyhaven.com/files/{ph_id}", timeout=30).json()
    url = info["Diffuse"][RES]["jpg"]["url"]
    img = Image.open(io.BytesIO(requests.get(url, timeout=120).content)).convert("RGB")
    if local == "pared_revoque":
        img = pintura_blanca(img)
    img.save(salida)
    print("ok", local, img.size)


