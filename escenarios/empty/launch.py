import mujoco
import mujoco.viewer
import os

# ==============================================================================
# CONFIGURACIÓN: Seleccionar escenario
# ==============================================================================
SCENARIO_FILE = "room.xml"  

# Leer el archivo de escenario
scenario_path = os.path.join("modelos", SCENARIO_FILE)
with open(scenario_path, "r", encoding="utf-8") as f:
    xml_content = f.read()

# Guardar directorio original
original_dir = os.getcwd()

try:
    # Cambiar a unitree_g1/ para que meshdir="assets" se resuelva correctamente
    os.chdir("unitree_g1")
    
    # Cargar el modelo
    model = mujoco.MjModel.from_xml_string(xml_content)
    
finally:
    # Restaurar directorio original (siempre, incluso si hay error)
    os.chdir(original_dir)

# Crear datos de simulación
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
