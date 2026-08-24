import os
import math
import mujoco
import mujoco.viewer
from PIL import Image
import keyboard

# Nombre de la carpeta dentro de escenarios/ que se quiere cargar
ESCENARIO = "habitacion_daniada"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITREE_DIR = os.path.join(PROJECT_ROOT, "unitree_g1")
ESCENARIO_DIR = os.path.join(PROJECT_ROOT, "escenarios", ESCENARIO)
CAPTURAS_DIR = os.path.join(PROJECT_ROOT, "capturas")
os.makedirs(CAPTURAS_DIR, exist_ok=True)

# Ruta relativa desde unitree_g1/ hacia el escenario, calculada automáticamente
rel_path = os.path.relpath(ESCENARIO_DIR, UNITREE_DIR).replace("\\", "/")

# Arma el XML combinado: robot (scene.xml) + escenario elegido
xml_content = f"""
<mujoco model="g1 con {ESCENARIO}">
  <include file="scene.xml"/>
  <include file="{rel_path}/escenario.xml"/>
</mujoco>
"""

# Carga la escena (robot + escenario). Se hace un chdir temporal a unitree_g1/
# para que el meshdir="assets" del robot resuelva bien las rutas de sus propias mallas.
original_dir = os.getcwd()

try:
    os.chdir(UNITREE_DIR)
    escena = mujoco.MjModel.from_xml_string(xml_content)
finally:
    os.chdir(original_dir)

data = mujoco.MjData(escena)

renderer = mujoco.Renderer(escena, 480, 640)

frame = 0
vel = 0.02
vel_giro = 0.03

yaw = 0.0

# Simulación con visor interactivo

with mujoco.viewer.launch_passive(escena, data) as viewer:
    viewer.cam.lookat[:] = data.qpos[:3]
    viewer.cam.distance = 3
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    while viewer.is_running():

        # Girar
        if keyboard.is_pressed("a"):
            yaw += vel_giro

        if keyboard.is_pressed("d"):
            yaw -= vel_giro

        # Avanzar / retroceder
        if keyboard.is_pressed("w"):
            data.qpos[0] += vel * math.cos(yaw)
            data.qpos[1] += vel * math.sin(yaw)

        if keyboard.is_pressed("s"):
            data.qpos[0] -= vel * math.cos(yaw)
            data.qpos[1] -= vel * math.sin(yaw)

        quat = [
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2)
        ]

        data.qpos[3:7] = quat

        mujoco.mj_forward(escena, data)
        mujoco.mj_step(escena, data)

        # Guardar una imagen cada 30 frames
        if frame % 30 == 0:
            renderer.update_scene(data, camera="head_camera")
            imagen = renderer.render()

            Image.fromarray(imagen).save(
                os.path.join(CAPTURAS_DIR, f"frame_{frame:05d}.png")
            )

        frame += 1
        viewer.sync()