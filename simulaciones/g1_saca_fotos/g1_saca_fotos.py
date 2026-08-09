import os
import math
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

MODELO = os.path.join(PROJECT_ROOT, "modelos", "detection.keras")

modelo = tf.keras.models.load_model(MODELO)

# Ruta relativa desde unitree_g1/ hacia el escenario, calculada automáticamente
rel_path = os.path.relpath(ESCENARIO_DIR, UNITREE_DIR).replace("\\", "/")

# Arma el XML combinado: robot (scene.xml) + escenario elegido
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

            img_resized = img.resize((224, 224))
            img_resized = np.array(img_resized)
            img_resized = np.expand_dims(img_resized, axis=0)

            pred_prob = modelo.predict(img_resized, verbose=0)[0][0]
            pred_label = "damaged" if pred_prob < 0.5 else "undamaged"

            print(f"Frame {frame}: {pred_label} (prob undamaged: {pred_prob:.4f})")

        frame += 1
        viewer.sync()