"""
Navegación autónoma del Unitree G1 hacia una secuencia de waypoints (x, y),
usando una política de locomoción RL exportada a ONNX.

Port a Python puro (mujoco.MjModel / mj_step, sin DDS ni SDK de Unitree) de
la lógica de g1_navigation_integration/reference_cpp_source/State_RLBase.cpp
(navigation_velocity_commands, navigation_gait_phase) validada previamente
con la pila de despliegue oficial de Unitree en C++.
"""

import math
import os
import sys
import time
import json
import io
import base64

# los modulos del algoritmo viven en carpetas y se importan por nombre, igual que antes
_RAIZ = os.path.dirname(os.path.abspath(__file__))
for _carpeta in ("navegacion", "inspeccion_danos"):
    sys.path.insert(0, os.path.join(_RAIZ, _carpeta))
import mapa_vivo
import localizacion
import slam_vivo
import plano
import exploracion
import cobertura
import inspeccion
import detector_danos as danos
import inspeccion_fina as fina
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml
import requests
from PIL import Image
import threading

# Agregar simulaciones/g1_conectado al path para importar calcular_room_layout
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "simulaciones", "g1_conectado"))
from extraer_dimensiones_CONECTADO import calcular_room_layout

# Backend URLs
BACKEND_DETECTIONS_URL = "http://127.0.0.1:8000/detections"
BACKEND_ROOM_URL = "http://127.0.0.1:8000/room"
BACKEND_MISSION_URL = "http://127.0.0.1:8000/mission"
BACKEND_MISSION_STATUS_URL = "http://127.0.0.1:8000/mission/status"
LAT_BACKEND = []   # (segundos que tardo el back en contestar cada foto, codigo HTTP)
# Escenario a cargar
SCENARIO = os.environ.get("G1_ESCENARIO", "edificio_informe")   # G1_ESCENARIO=escaneo_hospital carga un escaneo real
# Version del robot: 29 (G1 de 29 DOF, politica original) o 23 (G1 EDU Standard, 23 DOF).
# Por defecto es 23. Se puede cambiar con la variable de entorno G1_DOF (en PowerShell: $env:G1_DOF=29) o editando el 23 de abajo.
# Cada version usa su propio MJCF (unitree_g1/scene.xml o scene_23dof.xml) y su propia carpeta de politica
# (deploy.yaml + policy.onnx): g1_navigation_integration/policy o policy_23dof.
ROBOT_DOF = int(os.environ.get("G1_DOF", "23"))
if ROBOT_DOF not in (29, 23):
    raise SystemExit(f"G1_DOF tiene que ser 29 o 23 (vale {ROBOT_DOF})")
_POLICY_DIR = "g1_navigation_integration/policy" if ROBOT_DOF == 29 else "g1_navigation_integration/policy_23dof"
DEPLOY_YAML_PATH = f"{_POLICY_DIR}/deploy.yaml"
POLICY_ONNX_PATH = f"{_POLICY_DIR}/policy.onnx"
SCENE_XML = "scene.xml" if ROBOT_DOF == 29 else "scene_23dof.xml"
NAV_CONFIG_PATH = os.environ.get("G1_NAV_CONFIG", "g1_nav_config.yaml")

# Directorios para capturas
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTE_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "reporte")
REPORTE_IMAGENES_DIR = os.path.join(REPORTE_DIR, "images")
CAPTURAS_JSON = os.path.join(REPORTE_DIR, "capturas.json")
os.makedirs(REPORTE_IMAGENES_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "mapas"), exist_ok=True)   # mapas/ no se versiona: se crea aca

if os.path.exists(CAPTURAS_JSON):
    with open(CAPTURAS_JSON, "r", encoding="utf-8") as f:
        capturas = json.load(f)
else:
    capturas = []

FREE_JOINT_NAME = "floating_base_joint"
PELVIS_GYRO_SENSOR = "imu-pelvis-angular-velocity"

# Orden de las 29 articulaciones actuadas: idéntico al bloque <actuator> de
# unitree_g1/g1.xml y a los arrays de deploy.yaml (joint_ids_map identidad).
JOINT_NAMES_29 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Orden de las 23 articulaciones del G1 EDU Standard: el del MJCF con el que se entrena en mjlab
# (Unitree-G1-23Dof-Flat), idéntico al bloque <actuator> de unitree_g1/g1_23dof.xml y a los arrays de
# g1_navigation_integration/policy_23dof/deploy.yaml. Sin waist_roll, waist_pitch, wrist_pitch ni wrist_yaw.
JOINT_NAMES_23 = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
]

JOINT_NAMES = JOINT_NAMES_29 if ROBOT_DOF == 29 else JOINT_NAMES_23
N_JOINTS = len(JOINT_NAMES)
# vector de observacion: ang_vel(3) + gravedad(3) + comando(3) + fase(2) + pos(N) + vel(N) + ultima accion(N)
OBS_DIM = 3 + 3 + 3 + 2 + 3 * N_JOINTS   # 98 con 29 DOF, 80 con 23 DOF

GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])
LIDAR_BODY = "torso_link"
LIDAR_OFFSET = np.array([0.12, 0.0, 0.23])  # mismo punto que head_camera, en el marco del torso
LIDAR_HZ = 10.0
LIDAR_RANGE = (0.1, 20.0)  # m
LIDAR_GEOMGROUP = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)  # solo entorno, no el robot
RGBD_CAMERA = "head_camera"
RGBD_HZ = 2.0
GRABAR = True  # guarda la corrida en grabaciones/*.npz (entrada para el SLAM)
RGBD_SIZE = (240, 320)  # alto, ancho
RGBD_DEPTH_MAX = 10.0  # m; mas lejos se marca como invalido (inf)


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def guardar_captura(registro):
    """Guarda el registro de captura en capturas.json."""
    capturas.append(registro)
    with open(CAPTURAS_JSON, "w", encoding="utf-8") as f:
        json.dump(capturas, f, ensure_ascii=False, indent=2)


def planificar_destino(destino):
    """Ruta hasta 'destino' (x, y del mundo) sobre el ultimo mapa guardado en mapas/. Devuelve (waypoints, L, lo)."""
    import glob
    import planificador
    def es_de_este_escenario(f):
        m = np.load(f)
        return "escenario" in m.files and str(m["escenario"]) == SCENARIO
    mapas = [f for f in sorted(glob.glob(os.path.join(PROJECT_ROOT, "mapas", "*_mapa_estimado.npz"))) if es_de_este_escenario(f)]
    if not mapas:
        raise SystemExit(f"[g1_nav] no hay ningun mapa de {SCENARIO} en mapas/: graba una corrida y corre navegacion/mapa_ocupacion.py y navegacion/odometria_lidar.py")
    L, lo, _ = planificador.cargar_mapa(mapas[-1])
    off = LIDAR_OFFSET[:2]   # el mapa tiene su origen en el LiDAR del primer barrido; el robot nace en (0,0) del mundo
    wp, _, msg = planificador.planificar(L, lo, -off, np.asarray(destino, dtype=float) - off)
    if wp is None:
        raise SystemExit(f"[g1_nav] no se pudo planificar hasta {destino}: {msg}")
    print(f"[g1_nav] ruta planificada hasta {destino} sobre {os.path.basename(mapas[-1])}: {len(wp)} waypoints {msg}")
    return (wp + off).tolist(), L, lo


def cargar_mision(nav_config_path):
    """Intenta traer la misión (waypoints, arrival_radius, pause_duration)
    desde el backend (GET /mission). Si el backend no está corriendo o
    todavía no se definió ninguna misión, cae al yaml local (misma
    lógica que ya usa esta caminata cuando el backend no responde)."""
    try:
        resp = requests.get(BACKEND_MISSION_URL, timeout=3)
        resp.raise_for_status()
        mision = resp.json()
        if mision:
            print(f"[g1_nav] Misión obtenida del backend: {len(mision['waypoints'])} waypoints")
            return mision["waypoints"], mision["arrival_radius"], mision["pause_duration"]
        print("[g1_nav] El backend no tiene ninguna misión definida todavía, uso el yaml local.")
    except requests.exceptions.RequestException as e:
        print(f"[g1_nav] No se pudo obtener la misión del backend ({e}). Uso el yaml local.")

    nav_cfg = load_yaml(nav_config_path)
    return nav_cfg["waypoints"], nav_cfg["arrival_radius"], nav_cfg["pause_duration"]


def reportar_estado_mision(base_pos, yaw, nav_state, distance, waypoints_total, fallen):
    """Telemetría de baja frecuencia hacia el backend (no se llama en cada
    tick de física, solo en los mismos puntos donde ya se loguea por
    consola). Best-effort: si el backend no responde, no corta la caminata."""
    payload = {
        "x": float(base_pos[0]),
        "y": float(base_pos[1]),
        "yaw": float(yaw),
        "waypoint_idx": min(nav_state["current_idx"], waypoints_total - 1),
        "waypoints_total": waypoints_total,
        "distance_to_goal": float(distance),
        "waiting_at_waypoint": bool(nav_state["waiting_at_waypoint"]),
        "fallen": bool(fallen),
    }
    try:
        requests.post(BACKEND_MISSION_STATUS_URL, json=payload, timeout=3)
    except requests.exceptions.RequestException as e:
        if not getattr(reportar_estado_mision, "avisado", False):
            reportar_estado_mision.avisado = True
            print(f"[g1_nav] No se pudo reportar el estado de la misión al backend ({e}). No aviso de nuevo.")


def enviar_a_backend(registro, img_pil):
    """Envía la captura al backend de detecciones."""
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
    t0 = time.perf_counter()
    try:
        resp = requests.post(BACKEND_DETECTIONS_URL, json=payload, timeout=15)
        LAT_BACKEND.append((time.perf_counter() - t0, resp.status_code))
        print(f"  -> Captura enviada al backend (frame {registro['frame']}): respondio en {LAT_BACKEND[-1][0]:.2f} s (HTTP {resp.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"  -> No se pudo enviar al backend ({e}). ¿Está corriendo uvicorn?")


SIN_VENTANA = os.environ.get("G1_SIN_VENTANA") == "1"   # corre sin abrir la ventana de MuJoCo y sin esperar al reloj


class _SinVentana:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def is_running(self):
        return True

    def sync(self):
        pass


def build_model():
    """Construye el modelo MuJoCo con el escenario especificado.
    Similar a g1_saca_fotos_con_clasificacion_CONECTADO.py.
    """
    unitree_dir = os.path.join(PROJECT_ROOT, "unitree_g1")
    escenario_dir = os.path.join(PROJECT_ROOT, "escenarios", SCENARIO)
    
    rel_path = os.path.relpath(escenario_dir, unitree_dir).replace("\\", "/")
    
    # obstaculos sorpresa (no estan en el mapa): se declaran en g1_nav_config.yaml como [x, y, medio_ancho_x, medio_ancho_y, medio_alto]
    obstaculos_xml = ""
    for i, (ox, oy, hx, hy, hz) in enumerate(load_yaml(NAV_CONFIG_PATH).get("obstaculos") or []):
        obstaculos_xml += (f'  <worldbody><geom name="sorpresa_{i}" type="box" pos="{ox} {oy} {hz}" '
                           f'size="{hx} {hy} {hz}" rgba="0.8 0.3 0.2 1"/></worldbody>\n')

    xml_content = f"""
<mujoco model="g1 con {SCENARIO}">
  <include file="{SCENE_XML}"/>
  <include file="{rel_path}/escenario.xml"/>
{obstaculos_xml}</mujoco>
"""

    original_dir = os.getcwd()
    try:
        os.chdir(unitree_dir)
        model = mujoco.MjModel.from_xml_string(xml_content)
    finally:
        os.chdir(original_dir)

    data = mujoco.MjData(model)
    return model, data


def resolve_joint_addresses(model):
    """Direcciones explícitas por nombre (no se asume orden interno)."""
    qpos_adr = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES])
    dof_adr = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES])
    actuator_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINT_NAMES]
    )
    if np.any(actuator_ids < 0):
        missing = [n for n, i in zip(JOINT_NAMES, actuator_ids) if i < 0]
        raise ValueError(f"Actuadores no encontrados en el modelo: {missing}")
    free_qpos_adr = model.joint(FREE_JOINT_NAME).qposadr[0]
    free_dof_adr = model.joint(FREE_JOINT_NAME).dofadr[0]
    gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, PELVIS_GYRO_SENSOR)
    if gyro_id < 0:
        raise ValueError(f"Sensor no encontrado: {PELVIS_GYRO_SENSOR}")
    gyro_adr = model.sensor_adr[gyro_id]
    return {
        "qpos_adr": qpos_adr,
        "dof_adr": dof_adr,
        "actuator_ids": actuator_ids,
        "free_qpos_adr": free_qpos_adr,
        "free_dof_adr": free_dof_adr,
        "gyro_adr": gyro_adr,
    }


def apply_pd_gains(model, addr, stiffness, damping):
    """Sobreescribe las ganancias del actuador position implícito de MuJoCo
    para que reproduzcan el PD por articulación de deploy.yaml:
    fuerza = kp*(ctrl - qpos) - kd*qvel.
    """
    for i, aid in enumerate(addr["actuator_ids"]):
        model.actuator_gainprm[aid, 0] = stiffness[i]
        model.actuator_biasprm[aid, 1] = -stiffness[i]
        model.actuator_biasprm[aid, 2] = -damping[i]


def init_state(model, data, addr, default_joint_pos, stand_height=0.793):
    data.qpos[addr["free_qpos_adr"]: addr["free_qpos_adr"] + 3] = [0.0, 0.0, stand_height]
    data.qpos[addr["free_qpos_adr"] + 3: addr["free_qpos_adr"] + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[addr["qpos_adr"]] = default_joint_pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def get_base_pose(data, addr):
    pos = data.qpos[addr["free_qpos_adr"]: addr["free_qpos_adr"] + 3].copy()
    quat = data.qpos[addr["free_qpos_adr"] + 3: addr["free_qpos_adr"] + 7].copy()
    return pos, quat


def get_projected_gravity(quat):
    quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    res = np.zeros(3)
    mujoco.mju_rotVecQuat(res, GRAVITY_WORLD, quat_conj)
    return res


def get_base_ang_vel(data, addr):
    return data.sensordata[addr["gyro_adr"]: addr["gyro_adr"] + 3].copy()


def lidar_init(model):
    az = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    el = np.deg2rad(np.linspace(-7.0, 52.0, 16))
    A, E = np.meshgrid(az, el)
    dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1).reshape(-1, 3)
    return {"dirs": dirs, "n": len(dirs),
            "body_id": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LIDAR_BODY),
            "geomid": np.zeros(len(dirs), dtype=np.int32),
            "dist": np.zeros(len(dirs), dtype=np.float64)}


def lidar_scan(model, data, lid):
    R = data.xmat[lid["body_id"]].reshape(3, 3)
    origin = data.xpos[lid["body_id"]] + R @ LIDAR_OFFSET
    vec = (lid["dirs"] @ R.T).reshape(-1)
    mujoco.mj_multiRay(model, data, origin, vec, LIDAR_GEOMGROUP, 1, -1,
                       lid["geomid"], lid["dist"], None, lid["n"], LIDAR_RANGE[1])
    ranges = np.where(lid["geomid"] >= 0, lid["dist"], np.inf)
    ranges[ranges < LIDAR_RANGE[0]] = np.inf
    ranges[ranges > LIDAR_RANGE[1]] = np.inf
    return ranges.astype(np.float32), origin


def rgbd_init(model):
    r = mujoco.Renderer(model, RGBD_SIZE[0], RGBD_SIZE[1])
    # sombras/reflejos/cielo apagados: no aportan al sensor y son lo mas caro del render
    for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
              mujoco.mjtRndFlag.mjRND_SKYBOX, mujoco.mjtRndFlag.mjRND_HAZE):
        r.scene.flags[f] = 0
    return r


def rgbd_capture(data, r):
    r.disable_depth_rendering()
    r.update_scene(data, camera=RGBD_CAMERA)
    rgb = r.render().copy()
    opt = mujoco.MjvOption()
    opt.geomgroup[2] = 0  # esconde el robot (grupo visual) para que no aparezca en la profundidad
    r.enable_depth_rendering()
    r.update_scene(data, camera=RGBD_CAMERA, scene_option=opt)
    depth = r.render().copy()
    r.disable_depth_rendering()
    depth[depth > RGBD_DEPTH_MAX] = np.inf
    return rgb, depth

def guardar_calibracion(model, lid, ruta):
    """Escribe sensores.json: donde esta cada sensor respecto de la base e intrinsecas de la camara."""
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGBD_CAMERA)
    h, w = RGBD_SIZE
    f = (h / 2) / math.tan(math.radians(model.cam_fovy[cid]) / 2)
    el = np.unique(np.round(np.degrees(np.arcsin(lid["dirs"][:, 2])), 3))
    calib = {
        "tiempo": "segundos de simulacion (data.time)",
        "marco_base": LIDAR_BODY,
        "lidar": {
            "pos_en_base_m": LIDAR_OFFSET.tolist(),
            "orientacion": "mismos ejes que la base",
            "hz": LIDAR_HZ,
            "rango_m": list(LIDAR_RANGE),
            "rayos_por_giro": int(lid["n"] // len(el)),
            "elevaciones_grados": el.tolist(),
        },
        "rgbd": {
            "camara": RGBD_CAMERA,
            "pos_en_base_m": model.cam_pos[cid].tolist(),
            "quat_wxyz_en_base": model.cam_quat[cid].tolist(),
            "convencion": "MuJoCo: la camara mira por -Z, X a la derecha, Y arriba",
            "hz": RGBD_HZ,
            "alto": h, "ancho": w,
            "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
            "profundidad_max_m": RGBD_DEPTH_MAX,
        },
        "imu": {
            "torso": ["imu-torso-angular-velocity", "imu-torso-linear-acceleration"],
            "pelvis": ["imu-pelvis-angular-velocity", "imu-pelvis-linear-acceleration"],
        },
    }
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2, ensure_ascii=False)
    print(f"[g1_nav] calibracion de sensores guardada en {ruta}")


def guardar_grabacion(ruta, rec, waypoints):
    """Guarda lo grabado en un .npz. Todos los tiempos son segundos de simulacion (data.time)."""
    if not rec["t"]:
        return
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    np.savez_compressed(
        ruta,
        t=np.array(rec["t"]),  # una fila por paso de control (50 Hz)
        torso_pos=np.array(rec["torso_pos"]),  # posicion real del torso en el mundo (verdad terreno)
        torso_mat=np.array(rec["torso_mat"]).reshape(-1, 3, 3),  # orientacion real del torso
        imu=np.array(rec["imu"], dtype=np.float32),  # giro y aceleracion: torso(3+3) y pelvis(3+3)
        lidar_idx=np.array(rec["lidar_idx"]),  # a que fila de t corresponde cada barrido
        lidar=np.array(rec["lidar"]),  # distancias en m (inf = sin impacto)
        rgbd_idx=np.array(rec["rgbd_idx"]),  # a que fila de t corresponde cada imagen
        rgb=np.array(rec["rgb"]),
        depth=np.array(rec["depth"]),  # en m, float16 (inf = invalido)
        waypoints=np.array(waypoints),
        escenario=np.array(SCENARIO),
    )
    print(f"[g1_nav] grabacion guardada: {ruta} ({len(rec['t'])} pasos, {len(rec['lidar'])} barridos, {len(rec['rgb'])} imagenes)")


def nav_velocity_command(base_pos, quat, waypoints, nav_state, nav_params, vel_ranges,
                          arrival_radius, pause_duration, sim_time):
    """Port directo de navigation_velocity_commands (State_RLBase.cpp)."""
    idx = min(nav_state["current_idx"], len(waypoints) - 1)
    goal_x, goal_y = waypoints[idx]

    kp_lin = nav_params["kp_lin"]
    kp_ang = nav_params["kp_ang"]
    goal_tolerance = nav_params["goal_tolerance"]
    slow_radius = nav_params["slow_radius"]
    heading_deadband = nav_params["heading_deadband"]
    vy_gain = nav_params["vy_gain"]
    smoothing_alpha = nav_params["cmd_smoothing_alpha"]

    dx = goal_x - base_pos[0]
    dy = goal_y - base_pos[1]
    distance = math.hypot(dx, dy)

    w, x, y, z = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    vx_body = math.cos(yaw) * dx + math.sin(yaw) * dy
    vy_body = -math.sin(yaw) * dx + math.cos(yaw) * dy

    ramp = 1.0
    if slow_radius > goal_tolerance:
        ramp = clamp((distance - goal_tolerance) / (slow_radius - goal_tolerance), 0.0, 1.0)

    waypoints_remaining = (idx + 1) < len(waypoints)
    cmd = np.zeros(3)

    if distance <= arrival_radius and waypoints_remaining and nav_state["solo_pausa_final"]:
        nav_state["current_idx"] += 1   # waypoint intermedio de una ruta planificada: sigue de largo, sin pausa
    elif distance <= arrival_radius:
        if not nav_state["waiting_at_waypoint"]:
            nav_state["waiting_at_waypoint"] = True
            nav_state["arrived_time"] = sim_time
        elif waypoints_remaining:
            elapsed = sim_time - nav_state["arrived_time"]
            if elapsed >= pause_duration:
                nav_state["current_idx"] += 1
                nav_state["waiting_at_waypoint"] = False
        # cmd se queda en (0,0,0): pausado, o ya no quedan waypoints.
    else:
        nav_state["waiting_at_waypoint"] = False
        cmd[0] = clamp(kp_lin * vx_body * ramp, *vel_ranges["lin_vel_x"])
        cmd[1] = clamp(kp_lin * vy_body * ramp * vy_gain, *vel_ranges["lin_vel_y"])
        if distance > heading_deadband:
            target_yaw = math.atan2(dy, dx)
            yaw_error = wrap_to_pi(target_yaw - yaw)
            cmd[2] = clamp(kp_ang * yaw_error, *vel_ranges["ang_vel_z"])

    smoothed = smoothing_alpha * cmd + (1.0 - smoothing_alpha) * nav_state["smoothed_cmd"]
    nav_state["smoothed_cmd"] = smoothed

    return smoothed, distance, yaw


def navigation_gait_phase(period, step_dt, phase_state, cmd):
    """Port directo de navigation_gait_phase (State_RLBase.cpp)."""
    phase_state["global_phase"] = (phase_state["global_phase"] + step_dt / period) % 1.0
    cmd_norm = float(np.linalg.norm(cmd))
    obs = np.array([
        math.sin(phase_state["global_phase"] * 2.0 * math.pi),
        math.cos(phase_state["global_phase"] * 2.0 * math.pi),
    ])
    if cmd_norm < 0.1:
        obs[:] = 0.0
    return obs


def build_observation(data, addr, quat, joint_pos, joint_vel, default_joint_pos,
                       nav_cmd, gait_phase_obs, last_action):
    base_ang_vel = get_base_ang_vel(data, addr)
    projected_gravity = get_projected_gravity(quat)
    joint_pos_rel = joint_pos - default_joint_pos
    joint_vel_rel = joint_vel
    obs = np.concatenate([
        base_ang_vel, projected_gravity, nav_cmd, gait_phase_obs,
        joint_pos_rel, joint_vel_rel, last_action,
    ]).astype(np.float32)
    return obs.reshape(1, -1)


def main():
    # Enviar room layout al backend al iniciar
    room = None   # (si no se puede calcular el layout del escenario, plano.py arma los limites desde el mapa del SLAM)
    try:
        escenario_dir = os.path.join(PROJECT_ROOT, "escenarios", SCENARIO)
        room = calcular_room_layout(escenario_dir)
        r_room = requests.post(BACKEND_ROOM_URL, json=room, timeout=5)
        print(f"[g1_nav] Room layout enviado: {room['width']:.2f} m x {room['depth']:.2f} m (back: HTTP {r_room.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"[g1_nav] No se pudo enviar el room layout al backend ({e}). ¿Está corriendo uvicorn?")
    except Exception as e:
        print(f"[g1_nav] No se pudo calcular el room layout ({e}).")

    print(f"[g1_nav] robot: G1 de {ROBOT_DOF} DOF ({SCENE_XML}); politica: {POLICY_ONNX_PATH}")
    if not os.path.exists(POLICY_ONNX_PATH):
        raise SystemExit(f"[g1_nav] falta {POLICY_ONNX_PATH}: copia ahi el policy.onnx entrenado para {ROBOT_DOF} DOF (junto a su deploy.yaml)")
    deploy_cfg = load_yaml(DEPLOY_YAML_PATH)
    for _k in ("stiffness", "damping", "default_joint_pos"):
        if len(deploy_cfg[_k]) != N_JOINTS:
            raise ValueError(f"{DEPLOY_YAML_PATH}: '{_k}' tiene {len(deploy_cfg[_k])} valores y el robot tiene {N_JOINTS} articulaciones")

    stiffness = np.array(deploy_cfg["stiffness"], dtype=np.float64)
    damping = np.array(deploy_cfg["damping"], dtype=np.float64)
    default_joint_pos = np.array(deploy_cfg["default_joint_pos"], dtype=np.float64)
    action_scale = np.array(deploy_cfg["actions"]["JointPositionAction"]["scale"], dtype=np.float64)
    action_offset = np.array(deploy_cfg["actions"]["JointPositionAction"]["offset"], dtype=np.float64)
    vel_ranges = deploy_cfg["commands"]["base_velocity"]["ranges"]
    nav_params = deploy_cfg["observations"]["navigation_velocity_commands"]["params"]
    gait_period = deploy_cfg["observations"]["navigation_gait_phase"]["params"]["period"]
    control_dt = deploy_cfg["step_dt"]

    waypoints, arrival_radius, pause_duration = cargar_mision(NAV_CONFIG_PATH)
    destino = load_yaml(NAV_CONFIG_PATH).get("destino")   # si esta, manda sobre los waypoints de arriba
    L_mapa = lo_mapa = None
    if destino:
        waypoints, L_mapa, lo_mapa = planificar_destino(destino)

    model, data = build_model()
    if SCENARIO.startswith("escaneo_"):   # el piso del escaneo ya se ve: el piso cuadriculado de scene.xml sigue para pisar, pero invisible
        model.mat_rgba[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")][3] = 0.0
    addr = resolve_joint_addresses(model)
    apply_pd_gains(model, addr, stiffness, damping)
    init_state(model, data, addr, default_joint_pos)
    lid = lidar_init(model)
    lidar_every = max(1, round(1.0 / (LIDAR_HZ * control_dt)))
    rgbd_r = rgbd_init(model)
    rgbd_every = max(1, round(1.0 / (RGBD_HZ * control_dt)))
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGBD_CAMERA)
    f_cam = (RGBD_SIZE[0] / 2) / math.tan(math.radians(model.cam_fovy[cam_id]) / 2)
    off_mapa = LIDAR_OFFSET[:2]
    vivo = mapa_vivo.vivo_crear(L_mapa) if destino else None   # mapa de obstaculos en vivo (solo con ruta planificada)
    cfg_nav = load_yaml(NAV_CONFIG_PATH)
    loc = localizacion.loc_crear(L_mapa, lo_mapa, cfg_nav.get("gyro_sesgo_grados_s", 0.0), cfg_nav.get("gyro_ruido_rad_s", 0.0),
                                 bool(cfg_nav.get("corregir_rumbo"))) if destino else None   # pose estimada con LiDAR + mapa
    if loc is not None:
        print(f"[g1_nav] giroscopio simulado: sesgo {cfg_nav.get('gyro_sesgo_grados_s', 0.0)} grados/s, ruido {cfg_nav.get('gyro_ruido_rad_s', 0.0)} rad/s; "
              f"correccion de rumbo con LiDAR: {'SI' if loc['corregir'] else 'no'}")
    usar_estimada = bool(loc is not None and load_yaml(NAV_CONFIG_PATH).get("pose_estimada"))   # True: el control usa la pose estimada, no la real
    if usar_estimada:
        print("[g1_nav] navegando con la pose ESTIMADA (giroscopio + LiDAR + mapa), no con la del simulador")
    sl = None   # SLAM en vivo: mapa y pose propios, sin mapa previo (el origen es donde arranca el robot)
    if cfg_nav.get("slam"):
        if destino:
            raise SystemExit("[g1_nav] 'slam: true' y 'destino' no van juntos: con slam no hay mapa previo sobre el cual planificar")
        sl = slam_vivo.slam_crear(LIDAR_OFFSET[:2], cfg_nav.get("gyro_sesgo_grados_s", 0.0), cfg_nav.get("gyro_ruido_rad_s", 0.0),
                                  bool(cfg_nav.get("corregir_rumbo")))
        print(f"[g1_nav] SLAM en vivo: mapa propio desde cero; giroscopio con sesgo {cfg_nav.get('gyro_sesgo_grados_s', 0.0)} grados/s, "
              f"ruido {cfg_nav.get('gyro_ruido_rad_s', 0.0)} rad/s; correccion de rumbo con LiDAR: {'SI' if sl['corregir'] else 'no'}")
        usar_slam = bool(cfg_nav.get("pose_estimada"))
    else:
        usar_slam = False
    explorando = bool(cfg_nav.get("explorar"))   # el robot decide solo a donde ir: siempre la frontera alcanzable mas cercana
    ex = exploracion.explorar_crear()
    inspeccionando = bool(cfg_nav.get("inspeccionar"))   # despues de explorar, va a sacar fotos de todas las paredes
    ins = inspeccion.inspeccion_crear(cfg_nav)
    dn = danos.danos_crear(cfg_nav)   # detector de danos SIMULADO (el del back esta roto)
    if dn is not None:
        if dn.get("cnn") is not None:
            print(f"[g1_nav] detector de danos con la CNN del back ({dn['cnn']}), grilla {dn['grilla'][0]}x{dn['grilla'][1]}")
        else:
            print(f"[g1_nav] detector de danos simulado: {len(dn['reales'])} danos reales en el yaml, grilla {dn['grilla'][0]}x{dn['grilla'][1]}, prob. maxima {dn['p_max']}, falsos {dn['falsos']}")
    fi = fina.fina_crear(cfg_nav, dn, BACKEND_DETECTIONS_URL, REPORTE_IMAGENES_DIR)   # inspeccion fina de cada hallazgo (None si el yaml no tiene 'inspeccion_fina: true')
    if inspeccionando and not explorando:
        raise SystemExit("[g1_nav] 'inspeccionar: true' necesita 'explorar: true' (usa el mapa que arma la exploracion)")
    if explorando:
        if sl is None:
            raise SystemExit("[g1_nav] 'explorar: true' necesita 'slam: true'")
        usar_slam = True   # el mapa esta en el marco del arranque: el control tiene que usar la pose estimada
        waypoints[:] = [[0.0, 0.0]]
        print(f"[g1_nav] exploracion autonoma: tiempo maximo {cfg_nav.get('explorar_max_s', 900)} s")
    pl = plano.plano_crear(cfg_nav, room, BACKEND_ROOM_URL)   # mapa de paredes y posicion hacia el front (None si el yaml tiene 'plano_al_back: false')
    cob = cobertura.cob_crear(cfg_nav.get("alto_pared", 2.8)) if (sl is not None and cfg_nav.get("cobertura", True)) else None   # lo que la camara efectivamente inspecciono
    guardar_calibracion(model, lid, os.path.join(PROJECT_ROOT, "sensores.json"))
    rec = {"t": [], "torso_pos": [], "torso_mat": [], "imu": [], "lidar_idx": [], "lidar": [],
           "rgbd_idx": [], "rgb": [], "depth": []}
    imu_names = ("imu-torso-angular-velocity", "imu-torso-linear-acceleration",
                 "imu-pelvis-angular-velocity", "imu-pelvis-linear-acceleration")
    imu_adr = [model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)] for n in imu_names]

    # Renderer para captura de fotos desde la cámara del robot
    renderer = mujoco.Renderer(model, 480, 640)

    physics_dt = model.opt.timestep
    decimation = round(control_dt / physics_dt)
    print(f"[g1_nav] physics_dt={physics_dt}, control_dt={control_dt}, decimacion={decimation}")

    session = ort.InferenceSession(POLICY_ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_obs_dim = session.get_inputs()[0].shape[-1]
    if expected_obs_dim != OBS_DIM:
        raise ValueError(
            f"La red espera {expected_obs_dim} observaciones, el port construye {OBS_DIM} (robot de {ROBOT_DOF} DOF). "
            "Si la politica es de otro robot, cambia G1_DOF o POLICY_ONNX_PATH; si no, revisar el orden del vector de observaciones."
        )

    nav_state = {
        "current_idx": 0,
        "waiting_at_waypoint": False,
        "arrived_time": 0.0,
        "smoothed_cmd": np.zeros(3),
        "solo_pausa_final": bool(destino) or explorando,   # ruta planificada: pausa y foto solo en el ultimo waypoint
    }
    phase_state = {"global_phase": 0.0}
    last_action = np.zeros(N_JOINTS, dtype=np.float64)

    print(f"[g1_nav] waypoints={waypoints} arrival_radius={arrival_radius} pause_duration={pause_duration}")

    step_count = 0
    t_iter = time.perf_counter()
    t_next = time.perf_counter()
    t_log, sim_log = time.perf_counter(), 0.0
    n_log = 0        # cuantos segundos de log pasaron
    pct_ult = -1     # ultimo porcentaje de inspeccion impreso
    worst = {"iter": 0.0, "sync": 0.0}
    log_every = max(1, round(1.0 / control_dt))
    fallen_warned = False
    captured_at_waypoints = set()  # Para evitar capturas duplicadas en el mismo waypoint

    with (_SinVentana() if SIN_VENTANA else mujoco.viewer.launch_passive(model, data)) as viewer:
        while viewer.is_running():
            base_pos, quat = get_base_pose(data, addr)
            quat_nav = quat   # la politica sigue recibiendo la orientacion real (IMU); solo la navegacion usa la estimada
            if usar_estimada:
                base_pos, quat_nav = localizacion.loc_pose_base(loc, off_mapa, LIDAR_OFFSET)
            elif usar_slam:
                base_pos, quat_nav = localizacion.loc_pose_base(sl, np.zeros(2), LIDAR_OFFSET)
            joint_pos = data.qpos[addr["qpos_adr"]]
            joint_vel = data.qvel[addr["dof_adr"]]

            nav_cmd, distance, yaw = nav_velocity_command(
                base_pos, quat_nav, waypoints, nav_state, nav_params, vel_ranges,
                arrival_radius, pause_duration, data.time,
            )
            if inspeccionando and ins["vp"] is not None and nav_state["waiting_at_waypoint"]:   # llego al punto de vista: gira en el lugar hacia la pared
                err_yaw = inspeccion.inspeccion_orientar(ins, base_pos[:2], yaw, data.time)
                nav_cmd = nav_cmd.copy()
                nav_cmd[:2] = inspeccion.inspeccion_acercar(ins, base_pos[:2], yaw, nav_params["kp_lin"])   # ultimos centimetros hasta el punto exacto
                nav_cmd[2] = clamp(nav_params["kp_ang"] * err_yaw, *vel_ranges["ang_vel_z"])
                nav_state["smoothed_cmd"][:] = nav_cmd
            gait_phase_obs = navigation_gait_phase(gait_period, control_dt, phase_state, nav_cmd)

            obs = build_observation(
                data, addr, quat, joint_pos, joint_vel, default_joint_pos,
                nav_cmd, gait_phase_obs, last_action,
            )

            raw_action = session.run([output_name], {input_name: obs})[0].reshape(-1)
            target_q = raw_action * action_scale + action_offset
            data.ctrl[addr["actuator_ids"]] = target_q
            last_action = raw_action

            for _ in range(decimation):
                mujoco.mj_step(model, data)
            _t = time.perf_counter()
            viewer.sync()
            worst["sync"] = max(worst["sync"], time.perf_counter() - _t)

            # Capturar foto al llegar a cada waypoint (solo una vez por waypoint)
            wp_idx = nav_state["current_idx"]
            foto_inspeccion = inspeccionando and ins["listo"] and ins["vp"] is not None
            if (nav_state["waiting_at_waypoint"] and wp_idx not in captured_at_waypoints and not explorando) or foto_inspeccion:
                if foto_inspeccion:
                    etiqueta = f"fina_h{fi['act']['h']['id']}_{ins['vp']['fino']['paso']}" if ins['vp'].get('fino') else f"inspeccion_{ins['n']:03d}"
                else:
                    captured_at_waypoints.add(wp_idx)
                    etiqueta = f"waypoint_{wp_idx:02d}"

                renderer.update_scene(data, camera="head_camera")
                imagen = renderer.render()
                img = Image.fromarray(imagen)
                
                nombre_imagen = f"{etiqueta}_step_{step_count:06d}.png"
                img.save(os.path.join(REPORTE_IMAGENES_DIR, nombre_imagen))
                
                registro = {
                    "id": etiqueta,
                    "frame": step_count,
                    "timestamp": time.time(),
                    "x": float(base_pos[0]),
                    "y": float(base_pos[1]),
                    "image": f"images/{nombre_imagen}",
                }
                guardar_captura(registro)
                if not (foto_inspeccion and ins['vp'].get('fino')) and (not foto_inspeccion or cfg_nav.get('barrido_al_back', True)):   # las fotos de la inspeccion fina las manda inspeccion_fina.py, con la posicion del dano
                    threading.Thread(target=enviar_a_backend, args=(registro, img), daemon=True).start()
                if foto_inspeccion:
                    _, depth_foto = rgbd_capture(data, rgbd_r)   # imagen de profundidad del mismo instante que la foto
                    if dn is not None:
                        dn["img"] = imagen   # para el detector con la CNN (detector_cnn: true)
                    fino = fina.fina_foto(fi, ins, sl, data, cam_id, lid["body_id"], depth_foto, f_cam, img, registro, dn)   # None si no es una foto de la inspeccion fina
                    if fino is not None:
                        print(f"[g1_nav] {fino}")
                    if fino is None: cobertura.cob_foto_desde(cob, sl, data, cam_id, lid["body_id"], depth_foto, f_cam)   # las fotos de la inspeccion fina no suman a la cobertura del barrido
                    if fino is None: inspeccion.inspeccion_foto_hecha(ins, cob)   # (en la inspeccion fina libera el punto de vista fina.fina_foto)
                    if dn is not None and fino is None:
                        print(danos.dn_foto(dn, sl, data, cam_id, lid["body_id"], depth_foto, f_cam, ins["n"], data.time))
                    if fino is None: print(f"[g1_nav] inspeccion: foto {ins['n']} en ({base_pos[0]:.2f}, {base_pos[1]:.2f}), rumbo {np.degrees(yaw):.0f} grados, cobertura {cobertura.cob_resumen(cob)[0]:.0f}%")
                else:
                    print(f"[g1_nav] Foto capturada en waypoint {wp_idx}: ({base_pos[0]:.2f}, {base_pos[1]:.2f})")
                    reportar_estado_mision(base_pos, yaw, nav_state, distance, len(waypoints), fallen_warned)

            projected_gravity = get_projected_gravity(quat)
            if projected_gravity[2] > -0.3 and not fallen_warned:
                print("[g1_nav] ADVERTENCIA: el robot parece haberse caído o está muy inclinado.")
                fallen_warned = True
            elif projected_gravity[2] <= -0.3:
                fallen_warned = False

            if step_count % lidar_every == 0:
                ranges, lidar_origin = lidar_scan(model, data, lid)
                if vivo is not None:
                    mapa_vivo.vivo_agregar(vivo, lo_mapa, off_mapa, mapa_vivo.puntos_lidar(
                        data, lid["body_id"], lid["dirs"], lidar_origin, ranges, LIDAR_RANGE[1]))
                if loc is not None:
                    localizacion.loc_barrido(loc, lid["dirs"], ranges, LIDAR_RANGE[1], lidar_origin[:2] - off_mapa,
                                             data.xmat[lid["body_id"]].reshape(3, 3))
                if sl is not None:
                    slam_vivo.slam_barrido(sl, lid["dirs"], ranges, LIDAR_RANGE[1], lidar_origin,
                                           data.xmat[lid["body_id"]].reshape(3, 3), data.xpos[lid["body_id"]][:2])

            if step_count % rgbd_every == 0:
                rgb, depth = rgbd_capture(data, rgbd_r)
                if vivo is not None:
                    mapa_vivo.vivo_agregar(vivo, lo_mapa, off_mapa, mapa_vivo.puntos_profundidad(data, cam_id, depth, f_cam))
                if sl is not None and sl["n"] > 0:
                    slam_vivo.slam_bajos(sl, data, cam_id, lid["body_id"], depth, f_cam)
                if cob is not None and sl["n"] > 0 and not inspeccionando:   # en inspeccion solo cuentan las fotos sacadas orientado y quieto
                    cobertura.cob_foto_desde(cob, sl, data, cam_id, lid["body_id"], depth, f_cam)

            if loc is not None:   # el giroscopio se integra despues de usar el rumbo de este paso
                localizacion.loc_girar(loc, data.sensordata[imu_adr[0]:imu_adr[0] + 3], control_dt)
            if sl is not None:
                localizacion.loc_girar(sl, data.sensordata[imu_adr[0]:imu_adr[0] + 3], control_dt)

            if GRABAR:
                rec["t"].append(data.time)
                rec["torso_pos"].append(data.xpos[lid["body_id"]].copy())
                rec["torso_mat"].append(data.xmat[lid["body_id"]].copy())
                rec["imu"].append(np.concatenate([data.sensordata[a:a + 3] for a in imu_adr]))
                if step_count % lidar_every == 0:
                    rec["lidar_idx"].append(len(rec["t"]) - 1)
                    rec["lidar"].append(ranges)
                if step_count % rgbd_every == 0:
                    rec["rgbd_idx"].append(len(rec["t"]) - 1)
                    rec["rgb"].append(rgb)
                    rec["depth"].append(depth.astype(np.float16))

            plano.plano_tick(pl, sl, data.time, base_pos, yaw, ex["terminada"])   # posicion cada 0.5 s y paredes cada 'plano_cada_s', en un hilo
            step_count += 1
            if step_count % log_every == 0:
                n_log += 1
                worst["iter"] = worst["sync"] = 0.0
                if n_log % 30 == 0:   # velocidad real, una vez cada 30 s
                    _w = time.perf_counter()
                    print(f"[g1_nav] velocidad real = {(data.time - sim_log) / (_w - t_log):.2f}x")
                    t_log, sim_log = _w, data.time
                if vivo is not None:
                    nuevas = mapa_vivo.obstaculos_nuevos(vivo, L_mapa)
                    if nuevas.any():
                        wp_nuevos, msg_rp = mapa_vivo.replanificar(L_mapa, lo_mapa, off_mapa, nuevas, base_pos[:2], waypoints, nav_state["current_idx"], destino)
                        if wp_nuevos is not None:
                            waypoints[:] = wp_nuevos
                            nav_state["current_idx"] = 0
                            print(f"[g1_nav] ruta bloqueada por obstaculo nuevo: replanificada, {len(waypoints)} waypoints {msg_rp}")
                        elif msg_rp:
                            print(f"[g1_nav] ruta bloqueada por obstaculo nuevo y no hay otra: {msg_rp}")
                if sl is not None and sl["err"] and n_log % 10 == 0:   # error del SLAM, una vez cada 10 s
                    print(f"[g1_nav] slam: {slam_vivo.slam_resumen(sl)}")
                if explorando and sl["n"] > 10:
                    explorar_fin = ex["terminada"] or data.time > cfg_nav.get("explorar_max_s", 900)
                    if not explorar_fin:
                        wp_ex, msg_ex = exploracion.explorar_paso(sl, base_pos[:2], ex, waypoints, nav_state["current_idx"], nav_state["waiting_at_waypoint"])
                        if msg_ex and "corregida" not in msg_ex:
                            print(f"[g1_nav] exploracion: {msg_ex}")
                        if wp_ex is not None:
                            waypoints[:] = wp_ex.tolist()
                            nav_state["current_idx"] = 0
                            nav_state["waiting_at_waypoint"] = False
                    elif inspeccionando and not ins["terminada"] and data.time <= cfg_nav.get("inspeccion_max_s", 1200):
                        sl["congelado"] = True   # explorar ya dejo un mapa completo: de ahora en mas la posicion se calza contra esa referencia, sin que los errores se acumulen en ella
                        wp_in, msg_in = inspeccion.inspeccion_decidir(sl, cob, ins, base_pos[:2], data.cam_xpos[cam_id][2], data.time)
                        if msg_in:
                            print(f"[g1_nav] inspeccion: {msg_in}")
                        if wp_in is not None:
                            waypoints[:] = wp_in.tolist()
                            nav_state["current_idx"] = 0
                            nav_state["waiting_at_waypoint"] = False
                    elif fi is not None and not fi['terminada']:
                        sl['congelado'] = True
                        wp_fi, msgs_fi = fina.fina_decidir(fi, dn, sl, ins, base_pos[:2], data.cam_xpos[cam_id][2], f_cam, data.time)
                        for m_fi in msgs_fi:
                            print(f"[g1_nav] {m_fi}")
                        if wp_fi is not None:
                            waypoints[:] = wp_fi.tolist()
                            nav_state['current_idx'] = 0
                            nav_state['waiting_at_waypoint'] = False
                    else:
                        print(f"[g1_nav] fin a los {data.time:.0f} s de simulacion")
                        break
                if cob is not None and ((not explorando) or ex["terminada"]):   # el porcentaje solo tiene sentido en la fase de inspeccion
                    pct, cub, sin, vac = cobertura.cob_resumen(cob)
                    if int(pct) != pct_ult:
                        pct_ult = int(pct)
                        print(f"[g1_nav] INSPECCION {pct:.0f}% ({cub} de {cub + sin} tramos de pared vistos) a los {data.time:.0f} s")
                reportar_estado_mision(base_pos, yaw, nav_state, distance, len(waypoints), fallen_warned)

            worst["iter"] = max(worst["iter"], time.perf_counter() - t_iter)
            t_next += control_dt
            resto = t_next - time.perf_counter()
            if resto > 0:
                if not SIN_VENTANA:
                    time.sleep(resto)
            else:
                t_next = time.perf_counter()  # si se atrasa, no intenta recuperar de golpe
            t_iter = time.perf_counter()

    if pl is not None:
        print(plano.plano_final(pl, sl))   # el mapa final, para que el front lo tenga completo
    if sl is not None and sl["err"]:
        base_slam = os.path.join(PROJECT_ROOT, "mapas", f"slam_{time.strftime('%Y%m%d_%H%M%S')}")
        slam_vivo.slam_guardar(sl, base_slam, SCENARIO, waypoints)
        print(f"[g1_nav] mapa propio guardado: {base_slam}.png (celeste = estimado, verde = real) y .npz")
        if cob is not None:
            L_f, lo_f = slam_vivo.slam_recortar(sl)
            cobertura.cob_dibujar(cob, L_f, lo_f, np.array(sl["est"]), base_slam + "_cobertura.png")
            print(f"[g1_nav] cobertura final: {cobertura.cob_texto(cob)}; imagen: {base_slam}_cobertura.png")
            print(f"[g1_nav] cobertura por zona: {cobertura.cob_texto_zonas(cob, L_f, lo_f)}")
            print("[g1_nav] cobertura por altura: " + ", ".join(f"{a:.1f}-{b:.1f} m: {p:.0f}%" for a, b, p in cobertura.cob_por_altura(cob)))
            if inspeccionando:
                print(f"[g1_nav] inspeccion: {inspeccion.inspeccion_texto(ins, cob)}")
            if dn is not None:
                print(danos.dn_cierre(dn, sl, data.cam_xpos[cam_id][2], f_cam, base_slam))
                if fi is not None:
                    print(fina.fina_cierre(fi, dn, sl, base_slam))
            if LAT_BACKEND:
                seg = [x[0] for x in LAT_BACKEND]
                print(f"[g1_nav] back: {len(seg)} fotos contestadas, tardo en promedio {sum(seg) / len(seg):.2f} s (maximo {max(seg):.2f} s); con error HTTP: {sum(1 for x in LAT_BACKEND if x[1] != 200)}")

    if vivo is not None:
        from mapa_ocupacion import dibujar
        nuevas = mapa_vivo.obstaculos_nuevos(vivo, L_mapa)
        png = os.path.join(PROJECT_ROOT, "mapas", f"vivo_{time.strftime('%Y%m%d_%H%M%S')}.png")
        dibujar(L_mapa, lo_mapa, np.array(loc["est"]), np.array(waypoints) - off_mapa, png,
                trayecto_ref=np.array(loc["real"]), nuevas=nuevas)   # celeste = estimado, verde = real
        print(f"[g1_nav] mapa en vivo guardado: {png} ({int(nuevas.sum())} celdas nuevas en magenta)")

    if GRABAR:
        ruta = os.path.join(PROJECT_ROOT, "grabaciones", f"corrida_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        guardar_grabacion(ruta, rec, waypoints)


if __name__ == "__main__":
    main()
