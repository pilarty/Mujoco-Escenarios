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


def build_policy_paths():
    """Prioriza politicas nuevas de imitation."""
    import glob

    imitation_paths = glob.glob("training/imitation/**/PPOJax_saved.pkl", recursive=True)
    imitation_paths.sort(key=os.path.getmtime, reverse=True)
    return unique_paths(imitation_paths)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Evalua una politica entrenada de Unitree G1.")
    parser.add_argument("--steps", type=int, default=200, help="Cantidad de steps de evaluacion.")
    parser.add_argument("--sleep", type=float, default=0.02, help="Pausa entre steps cuando hay render.")
    parser.add_argument("--no-render", action="store_true", help="Ejecuta la evaluacion sin abrir MuJoCo viewer.")
    parser.add_argument("--render-every", type=int, default=1, help="Renderiza cada N steps.")
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
            traj=load_local_trajectory(env_params.get("env_name", "UnitreeG1"), local_npz)
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
    
    policy_paths = build_policy_paths()
    
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
                break
            except Exception as e:
                print(f"⚠ No se pudo cargar {path}: {e}")
                continue
    
    if policy_path is None:
        print("✗ No se encontró ninguna política entrenada")
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
            episode_reward += reward
            
            # Mostrar progreso
            if step % 50 == 0:
                print(f"   Step {step}/{args.steps} | Ep.{episode} | Steps: {episode_steps} | Reward: {episode_reward:.2f}")
            
            # Reset si termina
            if done:
                print(f"\n   ✓ Episodio {episode} completado: {episode_steps} steps, reward {episode_reward:.2f}")
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result
                episode += 1
                episode_steps = 0
                episode_reward = 0
                if episode <= 10:  # Solo mostrar primeros 10 episodios
                    print(f"   Episodio {episode}")
            
            if render and args.sleep > 0:
                time.sleep(args.sleep)
        
        print("\n✓ Evaluación completada con POLÍTICA ENTRENADA")
        
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
