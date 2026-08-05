import os
import mujoco
import mujoco.viewer

# Nombre de la carpeta dentro de escenarios/ que se quiere cargar
ESCENARIO = "habitacion_daniada"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITREE_DIR = os.path.join(PROJECT_ROOT, "unitree_g1")
ESCENARIO_DIR = os.path.join(PROJECT_ROOT, "escenarios", ESCENARIO)

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

# Simulación con visor interactivo
with mujoco.viewer.launch_passive(escena, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(escena, data)
        viewer.sync()