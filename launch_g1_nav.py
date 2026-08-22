"""
Navegación autónoma del Unitree G1 hacia una secuencia de waypoints (x, y),
usando una política de locomoción RL exportada a ONNX.

Port a Python puro (mujoco.MjModel / mj_step, sin DDS ni SDK de Unitree) de
la lógica de g1_navigation_integration/reference_cpp_source/State_RLBase.cpp
(navigation_velocity_commands, navigation_gait_phase) validada previamente
con la pila de despliegue oficial de Unitree en C++.
"""

import math
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import yaml

SCENE_PATH = "unitree_g1/scene.xml"
DEPLOY_YAML_PATH = "g1_navigation_integration/policy/deploy.yaml"
POLICY_ONNX_PATH = "g1_navigation_integration/policy/policy.onnx"
NAV_CONFIG_PATH = "g1_nav_config.yaml"

FREE_JOINT_NAME = "floating_base_joint"
PELVIS_GYRO_SENSOR = "imu-pelvis-angular-velocity"

# Orden de las 29 articulaciones actuadas: idéntico al bloque <actuator> de
# unitree_g1/g1.xml y a los arrays de deploy.yaml (joint_ids_map identidad).
JOINT_NAMES = [
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

GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
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

    if distance <= arrival_radius:
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
    deploy_cfg = load_yaml(DEPLOY_YAML_PATH)
    nav_cfg = load_yaml(NAV_CONFIG_PATH)

    stiffness = np.array(deploy_cfg["stiffness"], dtype=np.float64)
    damping = np.array(deploy_cfg["damping"], dtype=np.float64)
    default_joint_pos = np.array(deploy_cfg["default_joint_pos"], dtype=np.float64)
    action_scale = np.array(deploy_cfg["actions"]["JointPositionAction"]["scale"], dtype=np.float64)
    action_offset = np.array(deploy_cfg["actions"]["JointPositionAction"]["offset"], dtype=np.float64)
    vel_ranges = deploy_cfg["commands"]["base_velocity"]["ranges"]
    nav_params = deploy_cfg["observations"]["navigation_velocity_commands"]["params"]
    gait_period = deploy_cfg["observations"]["navigation_gait_phase"]["params"]["period"]
    control_dt = deploy_cfg["step_dt"]

    waypoints = nav_cfg["waypoints"]
    arrival_radius = nav_cfg["arrival_radius"]
    pause_duration = nav_cfg["pause_duration"]

    model, data = build_model()
    addr = resolve_joint_addresses(model)
    apply_pd_gains(model, addr, stiffness, damping)
    init_state(model, data, addr, default_joint_pos)

    physics_dt = model.opt.timestep
    decimation = round(control_dt / physics_dt)
    print(f"[g1_nav] physics_dt={physics_dt}, control_dt={control_dt}, decimacion={decimation}")

    session = ort.InferenceSession(POLICY_ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_obs_dim = session.get_inputs()[0].shape[-1]
    if expected_obs_dim != 98:
        raise ValueError(
            f"La red espera {expected_obs_dim} observaciones, el port construye 98. "
            "Revisar el orden del vector de observaciones antes de continuar."
        )

    nav_state = {
        "current_idx": 0,
        "waiting_at_waypoint": False,
        "arrived_time": 0.0,
        "smoothed_cmd": np.zeros(3),
    }
    phase_state = {"global_phase": 0.0}
    last_action = np.zeros(29, dtype=np.float64)

    print(f"[g1_nav] waypoints={waypoints} arrival_radius={arrival_radius} pause_duration={pause_duration}")

    step_count = 0
    log_every = max(1, round(1.0 / control_dt))
    fallen_warned = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            base_pos, quat = get_base_pose(data, addr)
            joint_pos = data.qpos[addr["qpos_adr"]]
            joint_vel = data.qvel[addr["dof_adr"]]

            nav_cmd, distance, yaw = nav_velocity_command(
                base_pos, quat, waypoints, nav_state, nav_params, vel_ranges,
                arrival_radius, pause_duration, data.time,
            )
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
            viewer.sync()

            projected_gravity = get_projected_gravity(quat)
            if projected_gravity[2] > -0.3 and not fallen_warned:
                print("[g1_nav] ADVERTENCIA: el robot parece haberse caído o está muy inclinado.")
                fallen_warned = True
            elif projected_gravity[2] <= -0.3:
                fallen_warned = False

            step_count += 1
            if step_count % log_every == 0:
                wp_idx = min(nav_state["current_idx"], len(waypoints) - 1)
                print(
                    f"[g1_nav] pos=({base_pos[0]:.2f},{base_pos[1]:.2f}) yaw={yaw:.2f} "
                    f"goal={waypoints[wp_idx]} dist={distance:.2f} "
                    f"cmd=({nav_cmd[0]:.2f},{nav_cmd[1]:.2f},{nav_cmd[2]:.2f}) "
                    f"waiting={nav_state['waiting_at_waypoint']}"
                )

            time.sleep(control_dt)


if __name__ == "__main__":
    main()
