"""
Diagnostico del flujo oficial LocoMuJoCo para Unitree G1.

Valida la superficie minima antes de entrenar:
- entornos registrados: UnitreeG1 y MjxUnitreeG1
- creacion mediante RLFactory
- reset y step sin render

No descarga datasets y no abre el viewer de MuJoCo.
"""
import os
import ssl
import sys
import traceback

import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")

from loco_mujoco import RLFactory, get_registered_envs


TARGET_ENVS = ("UnitreeG1", "MjxUnitreeG1")


def reset_env(env):
    result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def infer_action_dim(env):
    if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
        return env.action_space.shape[0]
    if hasattr(env, "model") and hasattr(env.model, "nu"):
        return env.model.nu
    if hasattr(env, "_action_spec"):
        return len(env._action_spec)
    if hasattr(env, "action_spec"):
        return len(env.action_spec)
    raise AttributeError("No se pudo inferir la dimension de accion")


def step_env(env, action):
    result = env.step(action)
    if not isinstance(result, tuple):
        raise TypeError(f"env.step devolvio {type(result).__name__}, se esperaba tuple")
    if len(result) not in (4, 5):
        raise ValueError(f"env.step devolvio {len(result)} valores, se esperaban 4 o 5")
    return result


def diagnose_env(env_name):
    print(f"\nProbando {env_name}")
    print("-" * 70)

    env = None
    try:
        env = RLFactory.make(env_name=env_name, headless=True)
        print(f"  Creado: {type(env).__name__}")

        obs = reset_env(env)
        print(f"  Reset OK | obs shape: {getattr(obs, 'shape', 'N/A')}")

        action_dim = infer_action_dim(env)
        print(f"  Action dim: {action_dim}")

        result = step_env(env, np.zeros(action_dim, dtype=np.float32))
        reward = result[1]
        print(f"  Step OK | reward: {reward}")
        return True
    except Exception as exc:
        print(f"  Error: {exc}")
        traceback.print_exc()
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main():
    print("=" * 70)
    print("DIAGNOSTICO G1 - FLUJO OFICIAL LOCOMUJOCO")
    print("=" * 70)

    registered_envs = get_registered_envs()
    print("\n1. Registro de entornos")
    print("=" * 70)

    missing = []
    for env_name in TARGET_ENVS:
        available = env_name in registered_envs
        print(f"  {env_name}: {'disponible' if available else 'NO disponible'}")
        if not available:
            missing.append(env_name)

    if missing:
        print(f"\nFaltan entornos requeridos: {', '.join(missing)}")
        return 1

    print("\n2. Reset/step con RLFactory")
    print("=" * 70)
    results = {env_name: diagnose_env(env_name) for env_name in TARGET_ENVS}

    print("\n3. Resumen")
    print("=" * 70)
    for env_name, ok in results.items():
        print(f"  {env_name}: {'OK' if ok else 'FALLO'}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())