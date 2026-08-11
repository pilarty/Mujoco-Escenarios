"""
Fase 4: Evaluación de Política Entrenada
Visualiza el robot G1 caminando con la política entrenada
"""
import argparse
import ssl
import time
import os
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

ssl._create_default_https_context = ssl._create_unverified_context

from loco_mujoco import RLFactory, TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.task_factories import CustomDatasetConf
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.datasets.humanoids.LAFAN1.load import extend_motion
from loco_mujoco.smpl.retargeting import load_robot_conf_file


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def unique_paths(paths):
    seen = set()
    result = []
    for path in paths:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def resolve_policy_path(policy_path):
    if policy_path is None:
        return None
    if os.path.isdir(policy_path):
        return os.path.join(policy_path, "PPOJax_saved.pkl")
    return policy_path


def build_policy_paths(policy_path=None):
    """Prioriza politicas nuevas de imitation."""
    import glob

    explicit_policy_path = resolve_policy_path(policy_path)
    if explicit_policy_path:
        return [explicit_policy_path]

    imitation_paths = glob.glob("training/imitation/**/PPOJax_saved.pkl", recursive=True)
    imitation_paths.sort(key=os.path.getmtime, reverse=True)
    return unique_paths(imitation_paths)


def print_checkpoint_config(agent_conf):
    config = getattr(agent_conf, "config", None)
    if config is None or not hasattr(config, "experiment"):
        print("  Config guardada: no disponible")
        return

    print("  Config guardada:")
    fields = [
        ("env", "experiment.env_params.env_name"),
        ("task_factory", "experiment.task_factory.name"),
        ("local_npz", "experiment.task_factory.params.local_npz"),
        ("total_timesteps", "experiment.total_timesteps"),
        ("num_envs", "experiment.num_envs"),
        ("num_steps", "experiment.num_steps"),
        ("normalize_env", "experiment.normalize_env"),
        ("reward_type", "experiment.env_params.reward_type"),
        ("lr", "experiment.lr"),
    ]
    for label, key in fields:
        value = OmegaConf.select(config, key, default="N/A")
        print(f"    - {label}: {value}")


def collect_env_objects(env):
    objects = []
    queue = [env]
    seen = set()
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        objects.append(obj)
        for attr in ("env", "_env", "unwrapped", "base_env"):
            try:
                child = getattr(obj, attr, None)
            except Exception:
                child = None
            if child is not None and id(child) not in seen:
                queue.append(child)
    return objects


def get_mujoco_qpos(env):
    for obj in collect_env_objects(env):
        for data_attr in ("data", "_data"):
            data = getattr(obj, data_attr, None)
            qpos = getattr(data, "qpos", None)
            if qpos is not None and len(qpos) >= 3:
                return np.array(qpos, dtype=np.float64).copy()
        for sim_attr in ("sim", "_sim"):
            sim = getattr(obj, sim_attr, None)
            data = getattr(sim, "data", None)
            qpos = getattr(data, "qpos", None)
            if qpos is not None and len(qpos) >= 3:
                return np.array(qpos, dtype=np.float64).copy()
    return None


def get_env_dt(env):
    for obj in collect_env_objects(env):
        for attr in ("dt", "timestep"):
            value = getattr(obj, attr, None)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        model = getattr(obj, "model", None) or getattr(obj, "_model", None)
        opt = getattr(model, "opt", None)
        timestep = getattr(opt, "timestep", None)
        if timestep is not None and timestep > 0:
            return float(timestep)
    return None


def print_physics_metrics(root_positions, total_steps, env, fall_step, fall_height):
    if len(root_positions) < 2:
        print("   Métricas físicas: no se pudo acceder a qpos de MuJoCo")
        return

    positions = np.array(root_positions)
    xy_delta = positions[-1, :2] - positions[0, :2]
    xy_distance = float(np.linalg.norm(xy_delta))
    heights = positions[:, 2]
    print("   Métricas físicas:")
    print(f"   - Desplazamiento XY: {xy_distance:.3f} m (dx={xy_delta[0]:.3f}, dy={xy_delta[1]:.3f})")
    print(f"   - Altura base min/media/final: {np.min(heights):.3f} / {np.mean(heights):.3f} / {heights[-1]:.3f} m")
    if fall_step is None:
        print(f"   - Caída por altura (< {fall_height:.2f} m): no detectada")
    else:
        print(f"   - Caída por altura (< {fall_height:.2f} m): step {fall_step}")
    if total_steps > 0:
        print(f"   - Avance promedio: {xy_distance / total_steps:.5f} m/step")
    dt = get_env_dt(env)
    if dt and total_steps > 0:
        print(f"   - Velocidad XY promedio estimada: {xy_distance / (total_steps * dt):.3f} m/s")


def base_env_name(env_name):
    return env_name.replace("Mjx", "", 1) if isinstance(env_name, str) and env_name.startswith("Mjx") else env_name


def resolve_local_npz(local_npz):
    """Resolve a dataset path saved in an older checkpoint config."""
    candidates = []
    if os.path.isabs(local_npz):
        candidates.append(local_npz)
    else:
        candidates.extend([
            local_npz,
            os.path.join(PROJECT_ROOT, local_npz),
        ])

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    filename = os.path.basename(local_npz)
    for root, _, files in os.walk(os.path.join(PROJECT_ROOT, "training")):
        if filename in files:
            return os.path.join(root, filename)

    return local_npz


def load_local_trajectory(env_name, local_npz):
    local_npz = resolve_local_npz(local_npz)
    traj = Trajectory.load(local_npz)
    if not traj.data.is_complete:
        source_env_name = base_env_name(env_name)
        traj = extend_motion(
            source_env_name,
            load_robot_conf_file(source_env_name),
            traj,
            replace_qvel_with_finite_diff=False,
        )
    return traj


def load_local_trajectories(env_name, local_npz_paths):
    paths = local_npz_paths if isinstance(local_npz_paths, list) else [local_npz_paths]
    trajectories = [load_local_trajectory(env_name, path) for path in paths]
    if len(trajectories) == 1:
        return trajectories[0]
    return Trajectory.concatenate(trajectories)


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua una politica entrenada de Unitree G1.")
    parser.add_argument("--policy-path", default=None, help="Ruta explicita al PPOJax_saved.pkl o a su carpeta.")
    parser.add_argument("--steps", type=int, default=200, help="Cantidad de steps de evaluacion.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Pausa entre steps cuando hay render.")
    parser.add_argument("--no-render", action="store_true", help="Ejecuta la evaluacion sin abrir MuJoCo viewer.")
    parser.add_argument("--render-every", type=int, default=1, help="Renderiza cada N steps.")
    parser.add_argument("--fall-height", type=float, default=0.35, help="Altura de base bajo la cual se reporta caida.")
    parser.add_argument("--fall-grace-steps", type=int, default=20, help="Steps iniciales ignorados para detectar caida por altura.")
    return parser.parse_args()


def build_eval_env(agent_conf, render=True):
    """Reconstruye el ambiente desde la config del agente y prioriza MuJoCo."""
    config = getattr(agent_conf, "config", None)
    if config is None or not hasattr(config, "experiment"):
        return RLFactory.make("UnitreeG1", horizon=1000, headless=not render)

    eval_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    OmegaConf.set_struct(eval_config, False)

    env_params = OmegaConf.to_container(eval_config.experiment.env_params, resolve=True)
    env_name = env_params.get("env_name", "UnitreeG1")
    if isinstance(env_name, str) and env_name.startswith("Mjx"):
        env_params["env_name"] = env_name.replace("Mjx", "", 1)
    env_params["headless"] = not render
    if "goal_params" in env_params:
        env_params["goal_params"]["visualize_goal"] = False

    task_factory = eval_config.experiment.get("task_factory", {"name": "RLFactory", "params": {}})
    factory_name = task_factory.get("name", "RLFactory")
    factory_params = OmegaConf.to_container(task_factory.get("params", {}) or {}, resolve=True)
    local_npz = factory_params.pop("local_npz", None)
    if local_npz:
        factory_params["custom_dataset_conf"] = CustomDatasetConf(
            traj=load_local_trajectories(env_params.get("env_name", "UnitreeG1"), local_npz)
        )
    factory = TaskFactory.get_factory_cls(factory_name)
    return factory.make(**env_params, **factory_params)

def main():
    args = parse_args()
    render = not args.no_render

    print("=" * 70)
    print("FASE 4: EVALUACIÓN DE POLÍTICA ENTRENADA")
    print("=" * 70)
    
    # Cargar política entrenada
    print("\n1. CARGANDO POLÍTICA ENTRENADA")
    print("=" * 70)
    
    policy_paths = build_policy_paths(args.policy_path)
    
    policy_path = None
    agent_conf = None
    agent_state = None
    
    for path in policy_paths:
        if os.path.exists(path):
            try:
                agent_conf, agent_state = PPOJax.load_agent(path)
                policy_path = path
                print(f"✓ Política cargada desde: {path}")
                
                # Mostrar info del archivo
                size = os.path.getsize(path) / 1024
                mod_time = os.path.getmtime(path)
                from datetime import datetime
                mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                print(f"  Tamaño: {size:.1f} KB")
                print(f"  Última modificación: {mod_date}")
                print_checkpoint_config(agent_conf)
                break
            except Exception as e:
                print(f"⚠ No se pudo cargar {path}: {e}")
                continue
    
    if policy_path is None:
        print("✗ No se encontró ninguna política entrenada")
        if args.policy_path:
            print(f"\nRuta solicitada: {resolve_policy_path(args.policy_path)}")
        else:
            print("\nBusqué en:")
            for path in policy_paths[:10]:  # Mostrar solo los principales
                print(f"  • {path}")
        print("\nEjecuta primero uno de estos:")
        print("  • python training/train_g1_imitation.py --local-npz training/walk1_subject1.npz")
        return
    
    # Crear ambiente de evaluación (MuJoCo estándar para visualización)
    print("\n2. CREANDO AMBIENTE DE EVALUACIÓN")
    print("=" * 70)
    
    # Determinar qué tipo de modelo se cargó
    if "imitation" in policy_path:
        print("🎯 Modelo: Imitation learning con dataset oficial")
        print("   Flujo preferido: ImitationFactory + dataset mocap + PPOJax")
    else:
        print("🎯 Modelo: política entrenada")
    
    try:
        env = build_eval_env(agent_conf, render=render)
        mode = "MuJoCo visual" if render else "headless"
        print(f"\n✓ Ambiente de evaluación creado: {type(env).__name__} ({mode})")
    except Exception as e:
        print(f"✗ Error al crear ambiente: {e}")
        return
    
    # Evaluar política
    print("\n3. EVALUANDO POLÍTICA")
    print("=" * 70)
    if render:
        print("\n📺 Se abrirá una ventana de MuJoCo mostrando el robot")
        print("   caminando con la política entrenada.\n")
        print("Controles:")
        print("  • Click izquierdo + arrastrar: rotar cámara")
        print("  • Click derecho + arrastrar: mover cámara")
        print("  • IMPORTANTE: NO hagas scroll (causa el bug que viste antes)")
    else:
        print("\nEjecutando sin render para aislar simulacion/politica del viewer.")
    print(f"\nLa evaluación correrá automáticamente por {args.steps} steps")
    print("Presiona Ctrl+C si quieres detener antes\n")
    print("=" * 70)
    
    try:
        # Evaluación manual con la política REAL entrenada
        print("\n🚀 Iniciando evaluación con política entrenada...")
        
        # Preparar train_state con log_std muy bajo para acciones deterministas
        train_state = agent_state.train_state
        train_state = train_state.replace(
            params={**train_state.params, 
                   'log_std': np.ones_like(train_state.params['log_std']) * -10}
        )
        
        # Función de inferencia (igual que en play_policy de LocoMuJoCo)
        def get_action(obs_array):
            """Obtiene acción determinista de la política entrenada"""
            obs_batch = np.expand_dims(obs_array, axis=0)  # Añadir dimensión batch
            
            # Aplicar red neuronal
            y, updates = agent_conf.network.apply(
                {'params': train_state.params, 'run_stats': train_state.run_stats},
                obs_batch,
                mutable=["run_stats"]
            )
            
            pi, _ = y  # pi = distribución de acciones, _ = value function
            
            # Acción determinista (media de la distribución)
            action = np.array(pi.mode()[0])  # mode() para determinista, [0] para remover batch
            
            return action
        
        # Reset
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        
        total_steps = 0
        episode = 1
        episode_steps = 0
        episode_reward = 0
        episode_lengths = []
        episode_rewards = []
        root_positions = []
        fall_step = None
        qpos = get_mujoco_qpos(env)
        if qpos is not None:
            root_positions.append(qpos[:3])
        
        print(f"   Episodio {episode}")
        
        for step in range(args.steps):
            # Obtener acción de la política ENTRENADA
            action = get_action(obs)
            
            # Step
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            
            # Renderizar
            if render and args.render_every > 0 and step % args.render_every == 0:
                try:
                    env.render()
                except Exception as render_error:
                    print(f"   Aviso: render fallo en step {step}: {render_error}")
            
            total_steps += 1
            episode_steps += 1
            episode_reward += float(reward)
            qpos = get_mujoco_qpos(env)
            if qpos is not None:
                root_positions.append(qpos[:3])
                if fall_step is None and step >= args.fall_grace_steps and qpos[2] < args.fall_height:
                    fall_step = step
            
            # Mostrar progreso
            if step % 50 == 0:
                print(f"   Step {step}/{args.steps} | Ep.{episode} | Steps: {episode_steps} | Reward: {episode_reward:.2f}")
            
            # Reset si termina
            if done:
                print(f"\n   ✓ Episodio {episode} completado: {episode_steps} steps, reward {episode_reward:.2f}")
                episode_lengths.append(episode_steps)
                episode_rewards.append(episode_reward)
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                episode += 1
                episode_steps = 0
                episode_reward = 0
                if episode <= 10:  # Solo mostrar primeros 10 episodios
                    print(f"   Episodio {episode}")
            
            if render and args.sleep > 0:
                time.sleep(args.sleep)
        
        if episode_steps > 0:
            episode_lengths.append(episode_steps)
            episode_rewards.append(episode_reward)

        print("\n✓ Evaluación completada con POLÍTICA ENTRENADA")
        if episode_lengths:
            print("\n📊 Métricas de evaluación:")
            print(f"   Episodios observados: {len(episode_lengths)}")
            print(f"   Largo medio: {np.mean(episode_lengths):.1f} steps")
            print(f"   Largo mínimo/máximo: {min(episode_lengths)} / {max(episode_lengths)} steps")
            print(f"   Reward medio: {np.mean(episode_rewards):.2f}")
            print_physics_metrics(root_positions, total_steps, env, fall_step, args.fall_height)
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Evaluación detenida por el usuario")
    except Exception as e:
        print(f"\n✗ Error durante evaluación: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cerrar ambiente limpiamente
        try:
            if 'env' in locals():
                env.close()
        except:
            pass
    
    print("\n" + "=" * 70)
    print("✓ FASE 4 COMPLETADA")
    print("=" * 70)
    print("\n🎯 RESUMEN:")
    print("   Has completado el pipeline completo:")
    print("   ✓ Fase 1: Instalación")
    print("   ✓ Fase 2: Visualización")
    print("   ✓ Fase 3: Entrenamiento")
    print("   ✓ Fase 4: Evaluación")
    print("\n💡 PRÓXIMOS PASOS (opcionales):")
    print("   • Re-entrenar con más timesteps para mejor calidad")
    print("   • Integrar en tu proyecto (experiments/02_locomotion_g1.py)")
    print("   • Ajustar hiperparámetros para mejor desempeño")


if __name__ == "__main__":
    main()
