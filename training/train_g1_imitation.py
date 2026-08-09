"""
Entrenamiento oficial por imitacion para Unitree G1 con LocoMuJoCo.

Este script sigue el camino recomendado por LocoMuJoCo para humanoides con
datasets de motion capture: ImitationFactory + MimicReward + GoalTrajMimic +
PPOJax. Por defecto corre un smoke test corto para validar el pipeline antes de
invertir tiempo en entrenamientos largos.
"""
import argparse
import os
import ssl
import sys
import time
import traceback
import warnings
from datetime import datetime

import jax
from omegaconf import OmegaConf

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

from loco_mujoco import ImitationFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import CustomDatasetConf, DefaultDatasetConf, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
from loco_mujoco.smpl.retargeting import load_robot_conf_file


FALLBACK_TASK_PROBES = [
    "walk",
    "run",
    "stepinplace",
    "stepinplace1",
    "squat",
]

LAFAN1_LOCOMOTION_TASKS = [
    "walk1_subject1",
    "walk1_subject2",
    "walk1_subject5",
    "walk2_subject1",
    "walk2_subject3",
    "walk2_subject4",
    "walk3_subject1",
    "walk3_subject2",
    "walk3_subject3",
    "walk3_subject4",
    "walk3_subject5",
    "walk4_subject1",
    "run1_subject2",
    "run1_subject5",
    "run2_subject1",
    "run2_subject4",
]

LAFAN1_ALIASES = {
    "walk": "walk1_subject1",
    "run": "run1_subject2",
}

SITES_FOR_MIMIC = [
    "upper_body_mimic",
    "left_hand_mimic",
    "left_foot_mimic",
    "right_hand_mimic",
    "right_foot_mimic",
]

_HF_SSL_CONFIGURED = False


def configure_huggingface_ssl():
    """Allow HuggingFace dataset downloads behind the corporate SSL proxy."""
    global _HF_SSL_CONFIGURED
    if _HF_SSL_CONFIGURED:
        return

    try:
        from urllib3.exceptions import InsecureRequestWarning

        warnings.simplefilter("ignore", InsecureRequestWarning)
    except Exception:
        pass

    try:
        import httpx
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import hf_request_event_hook

        def httpx_client_factory():
            return httpx.Client(
                verify=False,
                event_hooks={"request": [hf_request_event_hook]},
                follow_redirects=True,
                timeout=None,
            )

        set_client_factory(httpx_client_factory)
        _HF_SSL_CONFIGURED = True
        return
    except Exception:
        pass

    try:
        import requests
        from huggingface_hub import configure_http_backend

        def backend_factory():
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=backend_factory)
        _HF_SSL_CONFIGURED = True
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena Unitree G1 con imitation learning oficial de LocoMuJoCo.")
    parser.add_argument("--env", default="MjxUnitreeG1", choices=["UnitreeG1", "MjxUnitreeG1"])
    parser.add_argument("--dataset", default="auto", choices=["auto", "default", "lafan1"])
    parser.add_argument("--task", default=None, help="Task default/mocap a usar. Si se omite, se prueban candidatos.")
    parser.add_argument("--local-npz", default=None, help="Ruta a una trayectoria local .npz para evitar HuggingFace.")
    parser.add_argument("--timesteps", type=int, default=10_000, help="Timesteps totales. Default: smoke test corto.")
    parser.add_argument("--num-envs", type=int, default=4, help="Ambientes paralelos para smoke/local CPU.")
    parser.add_argument("--num-steps", type=int, default=10, help="Steps por rollout.")
    parser.add_argument("--output-dir", default="training/imitation", help="Directorio de salida del agente.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def normalize_lafan1_task(task):
    return LAFAN1_ALIASES.get(task, task).removesuffix(".csv").removesuffix(".npz")


def base_env_name(env_name):
    return env_name.replace("Mjx", "", 1) if env_name.startswith("Mjx") else env_name


def load_local_trajectory(env_name, local_npz):
    traj = Trajectory.load(local_npz)
    if not traj.data.is_complete:
        source_env_name = base_env_name(env_name)
        print("Trayectoria local incompleta: calculando datos derivados con MuJoCo...", flush=True)
        traj = extend_motion(
            source_env_name,
            load_robot_conf_file(source_env_name),
            traj,
            replace_qvel_with_finite_diff=False,
        )
    return traj


def make_config(args, dataset, task):
    task_factory_params = {}
    if dataset == "local":
        task_factory_params["local_npz"] = args.local_npz
    elif dataset == "lafan1":
        task_factory_params["lafan1_dataset_conf"] = {
            "dataset_name": normalize_lafan1_task(task),
        }
    else:
        task_factory_params["default_dataset_conf"] = {
            "task": task,
            "dataset_type": "mocap",
            "debug": False,
        }

    return OmegaConf.create({
        "experiment": {
            "task_factory": {
                "name": "ImitationFactory",
                "params": task_factory_params,
            },
            "env_params": {
                "env_name": args.env,
                "headless": True,
                "horizon": 1000,
                "goal_type": "GoalTrajMimic",
                "goal_params": {
                    "visualize_goal": True,
                },
                "reward_type": "MimicReward",
                "reward_params": {
                    "qpos_w_sum": 0.4,
                    "qvel_w_sum": 0.2,
                    "rpos_w_sum": 0.5,
                    "rquat_w_sum": 0.3,
                    "rvel_w_sum": 0.1,
                    "sites_for_mimic": SITES_FOR_MIMIC,
                },
            },
            "hidden_layers": [512, 256],
            "lr": 1e-4,
            "num_envs": args.num_envs,
            "num_steps": args.num_steps,
            "total_timesteps": args.timesteps,
            "update_epochs": 4,
            "num_minibatches": 2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "init_std": 0.2,
            "learnable_std": False,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "activation": "tanh",
            "anneal_lr": False,
            "weight_decay": 0.0,
            "normalize_env": True,
            "debug": False,
            "n_seeds": 1,
            "validation": {
                "active": False,
                "num_steps": 100,
                "num_envs": 100,
                "num": 10,
            },
        }
    })


def make_env(args, dataset, task):
    configure_huggingface_ssl()
    if dataset == "local":
        traj = load_local_trajectory(args.env, args.local_npz)
        return ImitationFactory.make(
            env_name=args.env,
            custom_dataset_conf=CustomDatasetConf(traj=traj),
            headless=True,
            horizon=1000,
            goal_type="GoalTrajMimic",
            goal_params={"visualize_goal": True},
            reward_type="MimicReward",
            reward_params={
                "qpos_w_sum": 0.4,
                "qvel_w_sum": 0.2,
                "rpos_w_sum": 0.5,
                "rquat_w_sum": 0.3,
                "rvel_w_sum": 0.1,
                "sites_for_mimic": SITES_FOR_MIMIC,
            },
        )

    if dataset == "lafan1":
        return ImitationFactory.make(
            env_name=args.env,
            lafan1_dataset_conf=LAFAN1DatasetConf(dataset_name=normalize_lafan1_task(task)),
            headless=True,
            horizon=1000,
            goal_type="GoalTrajMimic",
            goal_params={"visualize_goal": True},
            reward_type="MimicReward",
            reward_params={
                "qpos_w_sum": 0.4,
                "qvel_w_sum": 0.2,
                "rpos_w_sum": 0.5,
                "rquat_w_sum": 0.3,
                "rvel_w_sum": 0.1,
                "sites_for_mimic": SITES_FOR_MIMIC,
            },
        )

    dataset_conf = DefaultDatasetConf(task=task, dataset_type="mocap")
    return ImitationFactory.make(
        env_name=args.env,
        default_dataset_conf=dataset_conf,
        headless=True,
        horizon=1000,
        goal_type="GoalTrajMimic",
        goal_params={"visualize_goal": True},
        reward_type="MimicReward",
        reward_params={
            "qpos_w_sum": 0.4,
            "qvel_w_sum": 0.2,
            "rpos_w_sum": 0.5,
            "rquat_w_sum": 0.3,
            "rvel_w_sum": 0.1,
            "sites_for_mimic": SITES_FOR_MIMIC,
        },
    )


def make_first_available_env(args):
    if args.local_npz:
        task = os.path.splitext(os.path.basename(args.local_npz))[0]
        return "local", task, make_env(args, "local", task)

    if args.task:
        if args.dataset in ("default", "auto"):
            try:
                return "default", args.task, make_env(args, "default", args.task)
            except Exception as exc:
                if args.dataset == "default":
                    raise
                print(f"DefaultDataset no disponible para '{args.task}': {exc}", flush=True)
                if args.task not in LAFAN1_ALIASES and args.task not in LAFAN1_LOCOMOTION_TASKS:
                    raise
                print("Probando fallback oficial LAFAN1...", flush=True)

        lafan1_task = normalize_lafan1_task(args.task)
        return "lafan1", lafan1_task, make_env(args, "lafan1", lafan1_task)

    errors = []
    if args.dataset in ("default", "auto"):
        for task in FALLBACK_TASK_PROBES:
            print(f"Probando DefaultDataset: {task}", flush=True)
            try:
                return "default", task, make_env(args, "default", task)
            except Exception as exc:
                errors.append((f"default:{task}", exc))
                print(f"  No disponible: {exc}", flush=True)

    if args.dataset in ("lafan1", "auto"):
        for task in LAFAN1_LOCOMOTION_TASKS:
            print(f"Probando LAFAN1: {task}", flush=True)
            try:
                return "lafan1", task, make_env(args, "lafan1", task)
            except Exception as exc:
                errors.append((f"lafan1:{task}", exc))
                print(f"  No disponible: {exc}", flush=True)

    print("\nNo se pudo cargar ningun dataset candidato.", flush=True)
    for task, exc in errors:
        print(f"  - {task}: {exc}", flush=True)
    raise RuntimeError("No hay dataset default accesible para entrenar imitacion")


def main():
    args = parse_args()

    print("=" * 70, flush=True)
    print("ENTRENAMIENTO IMITACION - UNITREE G1", flush=True)
    print("=" * 70, flush=True)
    print(f"Entorno: {args.env}", flush=True)
    print(f"Dataset: {args.dataset}", flush=True)
    print(f"Timesteps: {args.timesteps:,}", flush=True)
    print(f"Salida: {args.output_dir}", flush=True)

    env = None
    try:
        print("\n1. Creando ambiente oficial con ImitationFactory", flush=True)
        print("=" * 70, flush=True)
        dataset, task, env = make_first_available_env(args)
        config = make_config(args, dataset, task)
        print(f"Dataset seleccionado: {dataset}:{task}", flush=True)

        print("\n2. Inicializando PPOJax", flush=True)
        print("=" * 70, flush=True)
        agent_conf = PPOJax.init_agent_conf(env, config)
        train_fn = PPOJax.build_train_fn(env, agent_conf)
        train_fn = jax.jit(train_fn)
        print("Agente y funcion de entrenamiento listos", flush=True)

        print("\n3. Entrenando", flush=True)
        print("=" * 70, flush=True)
        print(f"Inicio: {datetime.now().strftime('%H:%M:%S')}", flush=True)
        start = time.time()
        out = train_fn(jax.random.PRNGKey(args.seed))
        elapsed = time.time() - start

        agent_state = out["agent_state"]
        training_metrics = out["training_metrics"]
        final_return = float(training_metrics.mean_episode_return[-1])
        final_length = float(training_metrics.mean_episode_length[-1])

        print("\n4. Resultados", flush=True)
        print("=" * 70, flush=True)
        print(f"Tiempo: {elapsed:.1f}s", flush=True)
        print(f"Mean Episode Return: {final_return:.3f}", flush=True)
        print(f"Mean Episode Length: {final_length:.1f}", flush=True)

        output_dir = os.path.join(args.output_dir, dataset, task, f"{args.timesteps}_steps")
        os.makedirs(output_dir, exist_ok=True)
        save_path = PPOJax.save_agent(output_dir, agent_conf, agent_state)
        print(f"Agente guardado: {save_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.", flush=True)
        return 130
    except Exception as exc:
        print(f"\nError durante entrenamiento por imitacion: {exc}", flush=True)
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())