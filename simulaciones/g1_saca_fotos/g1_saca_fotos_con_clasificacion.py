import os
import json
import math
import time
import mujoco
import mujoco.viewer
from PIL import Image
import keyboard
import tensorflow as tf
import numpy as np

# Nombre de la carpeta dentro de escenarios/ que se quiere cargar
ESCENARIO = "habitacion_daniada"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNITREE_DIR = os.path.join(PROJECT_ROOT, "unitree_g1")
ESCENARIO_DIR = os.path.join(PROJECT_ROOT, "escenarios", ESCENARIO)
CAPTURAS_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "capturas")
os.makedirs(CAPTURAS_DIR, exist_ok=True)

# Carpeta que se va a servir para la interfaz web (JSON + fotos de daño)
REPORTE_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "reporte")
REPORTE_IMAGENES_DIR = os.path.join(REPORTE_DIR, "images")
DETECTIONS_JSON = os.path.join(REPORTE_DIR, "detections.json")
os.makedirs(REPORTE_IMAGENES_DIR, exist_ok=True)

# Se levanta la lista de detecciones ya existentes, si las hay, para no
# perderlas si se corta y se vuelve a correr la simulación.
if os.path.exists(DETECTIONS_JSON):
    with open(DETECTIONS_JSON, "r", encoding="utf-8") as f:
        detecciones = json.load(f)
else:
    detecciones = []


def guardar_deteccion(registro):
    """Agrega un registro a detections.json (lo reescribe entero cada vez,
    que para la cantidad de detecciones que maneja esto es más que suficiente)."""
    detecciones.append(registro)
    with open(DETECTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(detecciones, f, ensure_ascii=False, indent=2)

# Modelo 1: detecta si hay daño o no (damaged / undamaged)
MODELO_DETECCION = os.path.join(PROJECT_ROOT, "modelos", "detection.keras")
modelo_deteccion = tf.keras.models.load_model(MODELO_DETECCION)

# Modelo 2: clasifica el tipo de daño (multi-etiqueta, con umbrales por clase)
# Se carga desde ../models/ tal como en el notebook prediccion.ipynb
MODELO_CLASIFICACION = os.path.join(PROJECT_ROOT, "modelos", "modelo_classification.keras")
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, "modelos", "thresholds.json")

modelo_clasificacion = tf.keras.models.load_model(MODELO_CLASIFICACION)

with open(THRESHOLDS_PATH) as f:
    best_thresholds = json.load(f)

FINAL_CLASSES = [
    "Diagonal crack",
    "Concrete spalling",
    "Corrosion",
    "Shear crack",
    "Flexural/rocking crack",
    "Debris",
    "Detachment crack",
]

IMG_SIZE = (224, 224)


def clasificar_tipo_de_dano(img_pil):
    """
    Recibe una imagen PIL (ya redimensionada a 224x224) y devuelve
    una lista de tuplas (clase, probabilidad) para las clases que
    superan su umbral correspondiente, ordenadas de mayor a menor.
    """
    img_array = tf.keras.utils.img_to_array(img_pil)
    img_array = np.expand_dims(img_array, axis=0)

    probs = modelo_clasificacion.predict(img_array, verbose=0)[0]

    detectados = [
        (FINAL_CLASSES[i], probs[i])
        for i in range(len(FINAL_CLASSES))
        if probs[i] > best_thresholds[FINAL_CLASSES[i]]
    ]
    detectados.sort(key=lambda x: x[1], reverse=True)
    return detectados


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

            img_resized = img.resize(IMG_SIZE)
            img_array = np.array(img_resized)
            img_array_batch = np.expand_dims(img_array, axis=0)

            pred_prob = modelo_deteccion.predict(img_array_batch, verbose=0)[0][0]
            pred_label = "damaged" if pred_prob < 0.5 else "undamaged"

            print(f"Frame {frame}: {pred_label} (prob undamaged: {pred_prob:.4f})")

            # Si hay daño, se manda la misma imagen (ya en 224x224) al
            # segundo modelo para saber qué tipo de daño es.
            if pred_label == "damaged":
                detectados = clasificar_tipo_de_dano(img_resized)

                if detectados:
                    detalle = ", ".join(f"{c} ({p:.2%})" for c, p in detectados)
                else:
                    detalle = "sin tipo de daño específico detectado"

                print(f"  -> Tipo de daño: {detalle}")

                # Se guarda una copia de la foto en la carpeta que sirve la
                # interfaz web, y se agrega el registro a detections.json
                nombre_imagen = f"frame_{frame:05d}.png"
                img.save(os.path.join(REPORTE_IMAGENES_DIR, nombre_imagen))

                registro = {
                    "id": f"frame_{frame:05d}",
                    "frame": frame,
                    "timestamp": time.time(),
                    "x": float(data.qpos[0]),
                    "y": float(data.qpos[1]),
                    "image": f"images/{nombre_imagen}",
                    "detection_prob": float(1 - pred_prob),
                    "damage_types": [
                        {"class": c, "prob": float(p)} for c, p in detectados
                    ],
                }
                guardar_deteccion(registro)

        frame += 1
        viewer.sync()