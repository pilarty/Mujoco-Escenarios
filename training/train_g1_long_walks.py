"""
Entrenamiento largo de locomocion para Unitree G1 con multiples caminatas.

Usa varias trayectorias `walk*.npz` locales para evitar sobreajuste a una sola
secuencia. Los parametros por defecto son una primera prueba intermedia.
"""
import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_WALK_FILES = [
    "walk1_subject1.npz",
    "walk1_subject2.npz",
    "walk1_subject5.npz",
    "walk2_subject1.npz",
    "walk2_subject3.npz",
    "walk2_subject4.npz",
    "walk3_subject1.npz",
    "walk4_subject1.npz",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento largo con multiples trayectorias walk*.npz.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--all-walks", action="store_true", help="Usa todos los walk*.npz disponibles.")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    movements_dir = project_root / "training" / "Movements npz"
    trainer = project_root / "training" / "train_g1_imitation.py"

    if args.all_walks:
        local_npz = sorted(str(path) for path in movements_dir.glob("walk*.npz"))
    else:
        local_npz = [str(movements_dir / file_name) for file_name in DEFAULT_WALK_FILES]

    missing = [path for path in local_npz if not Path(path).exists()]
    if missing:
        print("Faltan trayectorias requeridas:")
        for path in missing:
            print(f"  - {path}")
        return 1

    command = [
        sys.executable,
        str(trainer),
        "--local-npz",
        *local_npz,
        "--timesteps",
        str(args.timesteps),
        "--num-envs",
        str(args.num_envs),
        "--num-steps",
        str(args.num_steps),
    ]
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())