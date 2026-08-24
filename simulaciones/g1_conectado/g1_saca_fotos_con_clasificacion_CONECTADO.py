import os
import sys
import json
import math
import time
import io
import base64
import mujoco
import mujoco.viewer
from PIL import Image
import keyboard
import requests     # pip install requests

BACKEND_URL = "http://localhost:8000/detections"
ROOM_URL = "http://localhost:8000/room"

# Nombre de la carpeta dentro de escenarios/ que se quiere cargar
ESCENARIO = "habitacion_daniada"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNITREE_DIR = os.path.join(PROJECT_ROOT, "unitree_g1")
ESCENARIO_DIR = os.path.join(PROJECT_ROOT, "escenarios", ESCENARIO)
CAPTURAS_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "capturas")
os.makedirs(CAPTURAS_DIR, exist_ok=True)

# Se agrega esta misma carpeta al path para poder importar la función
# de extraer_dimensiones_CONECTADO.py (vive junto a este script) sin
# duplicar su lógica acá.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extraer_dimensiones_CONECTADO import calcular_room_layout

# Se calcula y manda el layout de la habitación una sola vez al arrancar,
# para no tener que correr extraer_dimensiones_CONECTADO.py aparte.
try:
    _room = calcular_room_layout(ESCENARIO_DIR)
    requests.post(ROOM_URL, json=_room, timeout=5)
    print(f"Room layout enviado: {_room['width']:.2f} m x {_room['depth']:.2f} m")
except requests.exceptions.RequestException as e:
    print(f"No se pudo enviar el room layout al backend ({e}). ¿Está corriendo uvicorn?")
except Exception as e:
    print(f"No se pudo calcular el room layout ({e}).")

# Carpeta local con las capturas mandadas al backend (para debug)
REPORTE_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "reporte")
REPORTE_IMAGENES_DIR = os.path.join(REPORTE_DIR, "images")
CAPTURAS_JSON = os.path.join(REPORTE_DIR, "capturas.json")
os.makedirs(REPORTE_IMAGENES_DIR, exist_ok=True)

if os.path.exists(CAPTURAS_JSON):
    with open(CAPTURAS_JSON, "r", encoding="utf-8") as f:
        capturas = json.load(f)
else:
    capturas = []


def guardar_captura(registro):
    capturas.append(registro)
    with open(CAPTURAS_JSON, "w", encoding="utf-8") as f:
        json.dump(capturas, f, ensure_ascii=False, indent=2)


def enviar_a_backend(registro, img_pil):
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    imagen_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "frame": registro["frame"],
        "timestamp": registro["timestamp"],
        "x": registro["x"],
        "y": registro["y"],
        "imagen_base64": imagen_b64,
    }
    try:
        requests.post(BACKEND_URL, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"  -> No se pudo enviar al backend ({e}). ¿Está corriendo uvicorn?")


# Escena MuJoCo
rel_path = os.path.relpath(ESCENARIO_DIR, UNITREE_DIR).replace("\\", "/")

xml_content = f"""
<mujoco model="g1 con {ESCENARIO}">
  <include file="scene.xml"/>
  <include file="{rel_path}/escenario.xml"/>
</mujoco>
"""

original_dir = os.getcwd()
try:
    os.chdir(UNITREE_DIR)
    escena = mujoco.MjModel.from_xml_string(xml_content)
finally:
    os.chdir(original_dir)

data = mujoco.MjData(escena)

renderer = mujoco.Renderer(escena, 480, 640)

frame = 0
vel = 0.05
vel_giro = 0.03
yaw = 0.0

with mujoco.viewer.launch_passive(escena, data) as viewer:
    viewer.cam.lookat[:] = data.qpos[:3]
    viewer.cam.distance = 3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    while viewer.is_running():

        # Girar
        if keyboard.is_pressed("ctrl+left"):
            yaw += vel_giro

        if keyboard.is_pressed("ctrl+right"):
            yaw -= vel_giro

        # Avanzar / retroceder
        if keyboard.is_pressed("ctrl+up"):
            data.qpos[0] += vel * math.cos(yaw)
            data.qpos[1] += vel * math.sin(yaw)

        if keyboard.is_pressed("ctrl+down"):
            data.qpos[0] -= vel * math.cos(yaw)
            data.qpos[1] -= vel * math.sin(yaw)

        quat = [
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2)
        ]

        data.qpos[3:7] = quat

        mujoco.mj_step(escena, data)  # mj_forward no hace falta, mj_step ya lo incluye

        if frame % 300 == 0:
            renderer.update_scene(data, camera="head_camera")
            imagen = renderer.render()

            img = Image.fromarray(imagen)
            img.save(os.path.join(CAPTURAS_DIR, f"frame_{frame:05d}.png"))

            nombre_imagen = f"frame_{frame:05d}.png"
            img.save(os.path.join(REPORTE_IMAGENES_DIR, nombre_imagen))

            registro = {
                "id": f"frame_{frame:05d}",
                "frame": frame,
                "timestamp": time.time(),
                "x": float(data.qpos[0]),
                "y": float(data.qpos[1]),
                "image": f"images/{nombre_imagen}",
            }
            guardar_captura(registro)
            enviar_a_backend(registro, img)  # se manda la imagen completa, sin redimensionar

        frame += 1
        viewer.sync()
