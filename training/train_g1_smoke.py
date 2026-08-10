"""
Smoke test de entrenamiento de locomocion para Unitree G1.

Valida que el pipeline local funciona de punta a punta con una trayectoria:
.npz local -> ImitationFactory -> PPOJax -> guardado de politica.
"""
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    trainer = project_root / "training" / "train_g1_imitation.py"
    command = [
        sys.executable,
        str(trainer),
        "--local-npz",
        str(project_root / "training" / "Movements npz" / "walk1_subject1.npz"),
        "--timesteps",
        "10000",
        "--num-envs",
        "4",
        "--num-steps",
        "10",
    ]
    return subprocess.call(command, cwd=project_root)


if __name__ == "__main__":
    raise SystemExit(main())