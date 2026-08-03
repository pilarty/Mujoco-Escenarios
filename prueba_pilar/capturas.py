import mujoco
import numpy as np
from PIL import Image
import os

XML_PATH = "prueba_pilar/modelo.xml"
OUTPUT_DIR = "capturas_zoom"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=480, width=640)

cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)

cam.lookat = np.array([0.0, 0.0, 0.0])
cam.azimuth = 180
cam.elevation = -20

# Distintos niveles de zoom (de más cerca a más lejos)
distancias = [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]

for i, dist in enumerate(distancias):
    cam.distance = dist

    renderer.update_scene(data, camera=cam)
    img = renderer.render()

    filename = os.path.join(OUTPUT_DIR, f"zoom_{i:02d}_dist{dist}.png")
    Image.fromarray(img).save(filename)
    print(f"Guardada: {filename} (distance={dist})")

print(f"\nListo, {len(distancias)} capturas guardadas en '{OUTPUT_DIR}/'")