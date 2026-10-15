"""Renderiza las camaras inf_* del edificio_informe (robot + escenario) a PNG.
Uso:  python render_informe.py [ancho alto] [camara1 camara2 ...]
      (por defecto 1920 1080 y todas las vistas)
Salida: escenarios/edificio_informe/renders/<camara>.png
"""
import math
import os
import sys
import time

import mujoco
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(os.path.dirname(AQUI))
UNITREE = os.path.join(RAIZ, "unitree_g1")
SALIDA = os.path.join(AQUI, "renders")
os.makedirs(SALIDA, exist_ok=True)

args = sys.argv[1:]
W, H = 1920, 1080
if len(args) >= 2 and args[0].isdigit() and args[1].isdigit():
    W, H = int(args[0]), int(args[1])
    args = args[2:]

# camara: (pose del robot x, y, yaw, ocultar_techo)
VISTAS = {
    "inf_vista_aerea": ((0.0, 0.5, 1.57), True),
    "inf_planta": ((0.0, 0.5, 1.57), True),
    "inf_exterior": ((0.0, 1.5, 1.57), False),
    "inf_vestibulo": ((0.3, 3.2, 1.57), False),
    "inf_sala_a": ((-5.0, -0.8, 0.9), False),
    "inf_sala_a_grieta": ((-3.3, -1.7, 0.0), False),
    "inf_sala_b_techo": ((6.4, 1.2, 3.5), False),
    "inf_brecha": ((0.5, 5.4, 1.08), False),
    "inf_sala_c_humedad": ((-3.2, 7.4, 1.57), False),
    "head_camera": ((0.0, 2.2, 1.57), False),
}
if args:
    VISTAS = {k: v for k, v in VISTAS.items() if k in args}

rel = os.path.relpath(AQUI, UNITREE).replace("\\", "/")
xml = f"""<mujoco model="g1 edificio_informe">
  <include file="scene.xml"/>
  <include file="{rel}/escenario.xml"/>
</mujoco>"""

cwd = os.getcwd()
os.chdir(UNITREE)
try:
    model = mujoco.MjModel.from_xml_string(xml)
finally:
    os.chdir(cwd)
data = mujoco.MjData(model)
if model.nkey:
    mujoco.mj_resetDataKeyframe(model, data, 0)

renderer = mujoco.Renderer(model, H, W)
for nombre, (pose, ocultar_techo) in VISTAS.items():
    x, y, yaw = pose
    data.qpos[0], data.qpos[1] = x, y
    data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0 if ocultar_techo else 1
    t0 = time.time()
    renderer.update_scene(data, camera=nombre, scene_option=opt)
    img = renderer.render()
    Image.fromarray(img).save(os.path.join(SALIDA, nombre + ".png"))
    print(f"{nombre}: {time.time()-t0:.1f}s")
print("listo ->", SALIDA)
