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
    "pared_revoque": "plaster_grey_04",
    "piso_hormigon": "concrete_floor_worn_001",
    "piso_madera": "old_wood_floor",
    "ladrillo": "brick_wall_005",
    "mueble_madera": "dark_wood",
    "escombro": "concrete_debris",
}
RES = "2k"

for local, ph_id in TEXTURAS.items():
    salida = os.path.join(DEST, f"{local}.png")
    if os.path.exists(salida):
        print("ya existe", local)
        continue
    info = requests.get(f"https://api.polyhaven.com/files/{ph_id}", timeout=30).json()
    url = info["Diffuse"][RES]["jpg"]["url"]
    img = Image.open(io.BytesIO(requests.get(url, timeout=120).content)).convert("RGB")
    img.save(salida)
    print("ok", local, img.size)


