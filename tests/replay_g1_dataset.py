"""
Replay de datasets oficiales de LocoMuJoCo para Unitree G1.

Este script reemplaza la visualizacion con acciones aleatorias por una prueba
del flujo oficial basado en ImitationFactory y datasets de motion capture.
Por defecto usa UnitreeG1 para priorizar visualizacion en MuJoCo.
"""
import argparse
import os
import ssl
import sys
import traceback
import warnings
from pathlib import PurePosixPath

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

from loco_mujoco import ImitationFactory
from loco_mujoco.task_factories import CustomDatasetConf, DefaultDatasetConf, LAFAN1DatasetConf
from loco_mujoco.trajectory import Trajectory


DATASET_REPO = "robfiras/loco-mujoco-datasets"
DEFAULT_ENV = "UnitreeG1"

# Fallback probes only used when the HuggingFace file listing is unavailable.
# They are not treated as verified tasks until ImitationFactory loads them.
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

_HF_SSL_CONFIGURED = False


def configure_huggingface_ssl():
    """Allow HuggingFace downloads/listing behind the corporate SSL proxy."""
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


def base_env_name(env_name):
    """Return the dataset environment name used by LocoMuJoCo."""
    return env_name.replace("Mjx", "", 1) if env_name.startswith("Mjx") else env_name


def list_default_mocap_tasks(env_name):
    """List default mocap tasks available in the official dataset repo."""
    configure_huggingface_ssl()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return [], "huggingface_hub no esta instalado"

    dataset_env = base_env_name(env_name)
    prefix = f"DefaultDatasets/mocap/{dataset_env}/"

    listing_error = None
    try:
        repo_files = HfApi().list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    except Exception as exc:
        repo_files = []
        listing_error = f"no se pudo listar HuggingFace con HfApi ({exc})"

    if not repo_files:
        try:
            import requests

            api_url = (
                f"https://huggingface.co/api/datasets/{DATASET_REPO}"
                f"/tree/main/DefaultDatasets/mocap/{dataset_env}?recursive=true"
            )
            response = requests.get(api_url, timeout=30, verify=False)
            response.raise_for_status()
            repo_files = [item.get("path", "") for item in response.json()]
            listing_error = None
        except Exception as exc:
            secondary_error = f"no se pudo listar HuggingFace con requests sin SSL ({exc})"
            return [], listing_error or secondary_error

    tasks = []
    for file_name in repo_files:
        normalized_file_name = f"{prefix}{file_name}" if "/" not in file_name else file_name
        if normalized_file_name.startswith(prefix) and normalized_file_name.endswith(".npz"):
            tasks.append(PurePosixPath(file_name).stem)

    return sorted(set(tasks)), None


def print_lafan1_tasks():
    print("Tasks LAFAN1 oficiales disponibles localmente:")
    for task in LAFAN1_LOCOMOTION_TASKS:
        print(f"  - {task}")


def expected_cache_hint(env_name, dataset, task):
    dataset_env = base_env_name(env_name)
    cache_root = os.path.expanduser("~/.loco-mujoco-caches/datasets")
    if dataset == "lafan1":
        return os.path.join(cache_root, "lafan1", dataset_env, f"{normalize_lafan1_task(task)}.npz")
    return os.path.join(cache_root, "default", "mocap", dataset_env, f"{task}.npz")


def choose_task(env_name, dataset, requested_task, list_only):
    """Choose a dataset task without assuming unavailable names."""
    if dataset == "lafan1":
        print("\n1. Datasets oficiales")
        print("=" * 70)
        print_lafan1_tasks()
        return normalize_lafan1_task(requested_task or LAFAN1_LOCOMOTION_TASKS[0]) if not list_only else None

    tasks, listing_error = list_default_mocap_tasks(env_name)

    print("\n1. Datasets oficiales")
    print("=" * 70)
    if tasks:
        print(f"Tasks encontrados para {base_env_name(env_name)} ({len(tasks)}):")
        for task in tasks:
            print(f"  - {task}")
    else:
        print(f"No se pudo obtener listado verificado para {base_env_name(env_name)}.")
        if listing_error:
            print(f"Motivo: {listing_error}")

    if list_only:
        if not tasks:
            print("\nCandidatos no verificados para probar manualmente:")
            for task in FALLBACK_TASK_PROBES:
                print(f"  - {task}")
        print("\n")
        print_lafan1_tasks()
        return None

    if requested_task:
        if tasks and requested_task not in tasks:
            print(f"\nAdvertencia: '{requested_task}' no aparece en el listado verificado.")
            print("Se intentara cargar igual para confirmar con ImitationFactory.")
        return requested_task

    if tasks:
        locomotion_tasks = [task for task in tasks if any(token in task.lower() for token in ("walk", "run", "step"))]
        selected_task = locomotion_tasks[0] if locomotion_tasks else tasks[0]
        print(f"\nTask seleccionado automaticamente: {selected_task}")
        return selected_task

    print("\nSin listado remoto; se probaran candidatos comunes hasta que uno cargue.")
    return None


def normalize_lafan1_task(task):
    return LAFAN1_ALIASES.get(task, task).removesuffix(".csv").removesuffix(".npz")


def make_imitation_env(env_name, dataset, task, headless, local_npz=None):
    configure_huggingface_ssl()
    if local_npz:
        traj = Trajectory.load(local_npz)
        return ImitationFactory.make(
            env_name=env_name,
            custom_dataset_conf=CustomDatasetConf(traj=traj),
            headless=headless,
        )

    if dataset == "lafan1":
        return ImitationFactory.make(
            env_name=env_name,
            lafan1_dataset_conf=LAFAN1DatasetConf(dataset_name=normalize_lafan1_task(task)),
            headless=headless,
        )

    dataset_conf = DefaultDatasetConf(task=task, dataset_type="mocap")
    return ImitationFactory.make(
        env_name=env_name,
        default_dataset_conf=dataset_conf,
        headless=headless,
    )


def make_first_available_env(env_name, dataset, requested_task, headless, local_npz=None):
    if local_npz:
        task = os.path.splitext(os.path.basename(local_npz))[0]
        return "local", task, make_imitation_env(env_name, "local", task, headless, local_npz=local_npz)

    if requested_task:
        if dataset in ("default", "auto"):
            try:
                return "default", requested_task, make_imitation_env(env_name, "default", requested_task, headless)
            except Exception as exc:
                if dataset == "default":
                    print("\nNo se pudo acceder al dataset default remoto ni local.")
                    print(f"Cache local sugerido: {expected_cache_hint(env_name, 'default', requested_task)}")
                    raise
                print(f"DefaultDataset no disponible para '{requested_task}': {exc}")
                if requested_task not in LAFAN1_ALIASES and requested_task not in LAFAN1_LOCOMOTION_TASKS:
                    raise
                print("Probando fallback oficial LAFAN1...")

        lafan1_task = normalize_lafan1_task(requested_task)
        try:
            return "lafan1", lafan1_task, make_imitation_env(env_name, "lafan1", lafan1_task, headless)
        except Exception:
            print("\nNo se pudo acceder al dataset LAFAN1 remoto ni local.")
            print(f"Cache local sugerido: {expected_cache_hint(env_name, 'lafan1', lafan1_task)}")
            raise

    errors = []
    if dataset in ("default", "auto"):
        for task in FALLBACK_TASK_PROBES:
            print(f"Probando DefaultDataset candidato: {task}")
            try:
                return "default", task, make_imitation_env(env_name, "default", task, headless)
            except Exception as exc:
                errors.append((f"default:{task}", exc))
                print(f"  No disponible: {exc}")

    if dataset in ("lafan1", "auto"):
        for task in LAFAN1_LOCOMOTION_TASKS:
            print(f"Probando LAFAN1 candidato: {task}")
            try:
                return "lafan1", task, make_imitation_env(env_name, "lafan1", task, headless)
            except Exception as exc:
                errors.append((f"lafan1:{task}", exc))
                print(f"  No disponible: {exc}")

    print("\nNo se pudo cargar ningun candidato de dataset.")
    for task, exc in errors:
        print(f"  - {task}: {exc}")
    raise RuntimeError("No hay dataset default de G1 accesible con los candidatos probados")


def replay_trajectory(env, episodes, steps, render):
    if not hasattr(env, "play_trajectory"):
        raise AttributeError("El ambiente creado no expone play_trajectory")

    env.play_trajectory(
        n_episodes=episodes,
        n_steps_per_episode=steps,
        render=render,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lista y reproduce datasets oficiales de LocoMuJoCo para Unitree G1."
    )
    parser.add_argument("--env", default=DEFAULT_ENV, choices=["UnitreeG1", "MjxUnitreeG1"])
    parser.add_argument("--dataset", default="auto", choices=["auto", "default", "lafan1"])
    parser.add_argument("--task", default=None, help="Task de mocap a cargar; si se omite se elige una verificada.")
    parser.add_argument("--local-npz", default=None, help="Ruta a un .npz local de trayectoria LocoMuJoCo para evitar HuggingFace.")
    parser.add_argument("--list-only", action="store_true", help="Solo lista tasks oficiales disponibles.")
    parser.add_argument("--no-render", action="store_true", help="Carga el dataset sin abrir viewer.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    render = not args.no_render
    headless = not render

    print("=" * 70)
    print("REPLAY DATASET OFICIAL - UNITREE G1")
    print("=" * 70)
    print(f"Entorno: {args.env}")
    print(f"Dataset: {args.dataset}")
    print(f"Visualizacion MuJoCo: {'si' if render and args.env == 'UnitreeG1' else 'no'}")

    if args.local_npz:
        requested_task = os.path.splitext(os.path.basename(args.local_npz))[0]
        print("\n1. Dataset local")
        print("=" * 70)
        print(f"Archivo: {args.local_npz}")
    else:
        requested_task = choose_task(args.env, args.dataset, args.task, args.list_only)
    if args.list_only:
        return 0

    env = None
    try:
        print("\n2. Creando ambiente con ImitationFactory")
        print("=" * 70)
        dataset, task, env = make_first_available_env(
            args.env,
            args.dataset,
            requested_task,
            headless=headless,
            local_npz=args.local_npz,
        )
        print(f"Dataset cargado: {dataset}:{task}")
        print(f"Tipo de ambiente: {type(env).__name__}")

        print("\n3. Reproduciendo trayectoria")
        print("=" * 70)
        if render and args.env == "UnitreeG1":
            print("Se abrira el viewer de MuJoCo. Evita usar scroll si aparece el bug del viewer.")
        replay_trajectory(env, episodes=args.episodes, steps=args.steps, render=render)
        print("\nReplay completado correctamente.")
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrumpido por el usuario.")
        return 130
    except Exception as exc:
        print(f"\nError durante replay de dataset: {exc}")
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
