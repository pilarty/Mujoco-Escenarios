"""Carga el G1 dentro de un escenario de escaneo y guarda una vista general y lo que ve la camara de la cabeza.
Uso: python escaneos/prueba_integral/ver_escena.py escaneo_hospital"""
import os
import sys

import mujoco
import numpy as np
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
esc = sys.argv[1]
unitree = os.path.join(RAIZ, "unitree_g1")
rel = os.path.relpath(os.path.join(RAIZ, "escenarios", esc), unitree).replace("\\", "/")
xml = f'<mujoco><include file="scene.xml"/><include file="{rel}/escenario.xml"/></mujoco>'
antes = os.getcwd()
os.chdir(unitree)
m = mujoco.MjModel.from_xml_string(xml)
os.chdir(antes)
m.mat_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")][3] = 0.0   # como en launch_g1_nav.py
m.mat_reflectance[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")] = 0.0
d = mujoco.MjData(m)
d.qpos[2] = 0.793
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d, camera="head_camera")
fotos = [r.render()]
for az in (0, 120, 240):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.8]
    cam.distance, cam.azimuth, cam.elevation = 3.0, az, -25
    r.update_scene(d, camera=cam)
    fotos.append(r.render())
hoja = np.concatenate([np.concatenate(fotos[:2], 1), np.concatenate(fotos[2:], 1)], 0)
salida = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"vista_{esc}.jpg")
Image.fromarray(hoja).save(salida, quality=85)
print(salida)
