"""
Experimento 01: Saludo con el brazo derecho del Unitree G1

Este script demuestra cómo controlar las articulaciones del robot Unitree G1 en MuJoCo
para realizar un movimiento de saludo suave y natural.

Autor: Proyecto de simulación robótica
Fecha: 2026-07-09
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from typing import Dict, List, Tuple


# ===========================
# CONSTANTES DE CONFIGURACIÓN
# ===========================

# Ruta al archivo XML del escenario
SCENE_PATH = "unitree_g1/scene.xml"

# Frecuencia de simulación (Hz)
SIMULATION_FREQUENCY = 200

# Duraciones de cada fase del saludo (segundos)
STABILIZATION_TIME = 2.0      # Tiempo inicial para estabilizar la simulación
ARM_RAISE_TIME = 2.0          # Tiempo para levantar el brazo
WAVE_CYCLE_TIME = 0.8         # Duración de un ciclo completo de saludo
NUMBER_OF_WAVES = 3           # Cantidad de saludos (movimientos de lado a lado)
ARM_LOWER_TIME = 2.0          # Tiempo para bajar el brazo
FINAL_PAUSE_TIME = 2.0        # Tiempo final en posición de reposo

# Ángulos objetivo para las articulaciones del brazo derecho (radianes)
# Posición inicial (brazo relajado al costado)
ARM_INITIAL_ANGLES = {
    'right_shoulder_pitch': 0.2,    # Brazo ligeramente hacia adelante
    'right_shoulder_roll': -0.2,    # Brazo al costado del cuerpo
    'right_shoulder_yaw': 0.0,      # Sin rotación
    'right_elbow': 1.28,            # Codo doblado
    'right_wrist_roll': 0.0,        # Sin rotación de muñeca
    'right_wrist_pitch': 0.0,       # Muñeca recta
    'right_wrist_yaw': 0.0          # Sin rotación de muñeca
}

# Posición levantada (brazo en alto para saludar)
ARM_RAISED_ANGLES = {
    'right_shoulder_pitch': -1.2,   # Brazo levantado hacia adelante
    'right_shoulder_roll': -1.2,    # Brazo separado del cuerpo
    'right_shoulder_yaw': 0.0,      # Sin rotación
    'right_elbow': 1.8,             # Codo más doblado
    'right_wrist_roll': 0.0,        # Sin rotación
    'right_wrist_pitch': 0.0,       # Muñeca recta
    'right_wrist_yaw': 0.0          # Sin rotación
}

# Amplitud del movimiento de saludo (radianes)
WAVE_AMPLITUDE = 0.6  # Movimiento de la muñeca de lado a lado


# ===========================
# FUNCIONES DE UTILIDAD
# ===========================

def listar_articulaciones_y_actuadores(model: mujoco.MjModel) -> None:
    """
    Imprime información sobre todas las articulaciones y actuadores del modelo.
    
    Esta función es útil para entender la estructura del robot y verificar
    los nombres correctos de las articulaciones antes de controlarlas.
    
    Args:
        model: Modelo de MuJoCo cargado
    """
    print("\n" + "="*70)
    print("INFORMACIÓN DEL MODELO")
    print("="*70)
    
    # Listar todas las articulaciones
    print(f"\n📋 Total de articulaciones: {model.njnt}")
    print("\nNombres de articulaciones:")
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        joint_type = model.jnt_type[i]
        joint_range = model.jnt_range[i]
        print(f"  [{i:2d}] {joint_name:35s} (Tipo: {joint_type}, Rango: [{joint_range[0]:6.2f}, {joint_range[1]:6.2f}])")
    
    # Listar todos los actuadores
    print(f"\n🎮 Total de actuadores: {model.nu}")
    print("\nNombres de actuadores:")
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        joint_id = model.actuator_trnid[i, 0]
        joint_name = model.joint(joint_id).name if joint_id >= 0 else "N/A"
        print(f"  [{i:2d}] {actuator_name:35s} -> Controla: {joint_name}")
    
    print("="*70 + "\n")


def obtener_indices_actuadores(model: mujoco.MjModel, 
                               prefijos: List[str]) -> Dict[str, int]:
    """
    Obtiene los índices de los actuadores que coinciden con los prefijos dados.
    
    Esta función busca automáticamente los actuadores del brazo derecho sin
    necesidad de especificar índices manualmente.
    
    Args:
        model: Modelo de MuJoCo
        prefijos: Lista de prefijos de nombres de actuadores a buscar
        
    Returns:
        Diccionario mapeando nombre simplificado -> índice de actuador
    """
    indices = {}
    
    for i in range(model.nu):
        actuator_name = model.actuator(i).name
        
        for prefijo in prefijos:
            if prefijo in actuator_name:
                # Crear clave simplificada (sin "_joint")
                key = actuator_name.replace("_joint", "")
                indices[key] = i
                break
    
    return indices


def interpolar_lineal(inicio: float, fin: float, t: float) -> float:
    """
    Interpolación lineal simple entre dos valores.
    
    Args:
        inicio: Valor inicial
        fin: Valor final
        t: Parámetro de interpolación (0.0 = inicio, 1.0 = fin)
        
    Returns:
        Valor interpolado
    """
    return inicio + (fin - inicio) * t


def interpolar_suave(inicio: float, fin: float, t: float) -> float:
    """
    Interpolación suave usando una función coseno (ease-in-out).
    
    Esta función produce movimientos más naturales que la interpolación lineal,
    comenzando y terminando lentamente con velocidad máxima en el medio.
    
    Args:
        inicio: Valor inicial
        fin: Valor final
        t: Parámetro de interpolación (0.0 = inicio, 1.0 = fin)
        
    Returns:
        Valor interpolado suavemente
    """
    # Aplicar función coseno para suavizar
    t_suave = (1 - np.cos(t * np.pi)) / 2
    return inicio + (fin - inicio) * t_suave


def aplicar_control(model: mujoco.MjModel,
                   data: mujoco.MjData,
                   indices: Dict[str, int],
                   angulos: Dict[str, float]) -> None:
    """
    Aplica los ángulos objetivo a los actuadores especificados.
    
    Args:
        model: Modelo de MuJoCo
        data: Datos de simulación de MuJoCo
        indices: Diccionario de índices de actuadores
        angulos: Diccionario de ángulos objetivo para cada articulación
    """
    for key, angulo in angulos.items():
        if key in indices:
            data.ctrl[indices[key]] = angulo


# ===========================
# FUNCIONES DE MOVIMIENTO
# ===========================

def estabilizar_simulacion(model: mujoco.MjModel,
                          data: mujoco.MjData,
                          viewer: mujoco.viewer.Handle,
                          duracion: float) -> None:
    """
    Ejecuta la simulación sin movimientos adicionales para estabilizar el robot.
    
    Args:
        model: Modelo de MuJoCo
        data: Datos de simulación
        viewer: Visor de MuJoCo
        duracion: Tiempo de estabilización en segundos
    """
    print(f"⏳ Estabilizando simulación ({duracion:.1f}s)...")
    
    dt = 1.0 / SIMULATION_FREQUENCY
    num_pasos = int(duracion * SIMULATION_FREQUENCY)
    
    for _ in range(num_pasos):
        if not viewer.is_running():
            break
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
    
    print("✓ Simulación estabilizada")


def mover_brazo(model: mujoco.MjModel,
               data: mujoco.MjData,
               viewer: mujoco.viewer.Handle,
               indices: Dict[str, int],
               angulos_inicio: Dict[str, float],
               angulos_fin: Dict[str, float],
               duracion: float,
               descripcion: str) -> None:
    """
    Mueve el brazo de una posición inicial a una final de forma suave.
    
    Args:
        model: Modelo de MuJoCo
        data: Datos de simulación
        viewer: Visor de MuJoCo
        indices: Diccionario de índices de actuadores
        angulos_inicio: Ángulos iniciales de las articulaciones
        angulos_fin: Ángulos finales de las articulaciones
        duracion: Tiempo del movimiento en segundos
        descripcion: Descripción del movimiento para imprimir
    """
    print(f"🤖 {descripcion}...")
    
    dt = 1.0 / SIMULATION_FREQUENCY
    num_pasos = int(duracion * SIMULATION_FREQUENCY)
    
    for paso in range(num_pasos):
        if not viewer.is_running():
            break
        
        # Calcular progreso (0.0 a 1.0)
        t = paso / num_pasos
        
        # Interpolar cada ángulo
        angulos_actuales = {}
        for key in angulos_inicio.keys():
            angulos_actuales[key] = interpolar_suave(
                angulos_inicio[key],
                angulos_fin[key],
                t
            )
        
        # Aplicar control
        aplicar_control(model, data, indices, angulos_actuales)
        
        # Avanzar simulación
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(dt)
    
    print(f"✓ {descripcion} completado")


def ejecutar_saludo(model: mujoco.MjModel,
                   data: mujoco.MjData,
                   viewer: mujoco.viewer.Handle,
                   indices: Dict[str, int],
                   num_ciclos: int,
                   duracion_ciclo: float,
                   amplitud: float) -> None:
    """
    Ejecuta el movimiento de saludo moviendo la muñeca de lado a lado.
    
    Args:
        model: Modelo de MuJoCo
        data: Datos de simulación
        viewer: Visor de MuJoCo
        indices: Diccionario de índices de actuadores
        num_ciclos: Número de veces que se repite el saludo
        duracion_ciclo: Duración de un ciclo completo (ida y vuelta)
        amplitud: Amplitud del movimiento en radianes
    """
    print(f"👋 Ejecutando saludo ({num_ciclos} ciclos)...")
    
    dt = 1.0 / SIMULATION_FREQUENCY
    num_pasos_por_ciclo = int(duracion_ciclo * SIMULATION_FREQUENCY)
    
    for ciclo in range(num_ciclos):
        print(f"   Ciclo {ciclo + 1}/{num_ciclos}")
        
        for paso in range(num_pasos_por_ciclo):
            if not viewer.is_running():
                break
            
            # Calcular ángulo de saludo usando función seno
            # Esto crea un movimiento suave de lado a lado
            t = paso / num_pasos_por_ciclo
            angulo_yaw = amplitud * np.sin(2 * np.pi * t)
            
            # Mantener ángulos base del brazo levantado
            angulos = ARM_RAISED_ANGLES.copy()
            # Modificar solo la rotación de la muñeca para el saludo
            angulos['right_wrist_yaw'] = angulo_yaw
            
            # Aplicar control
            aplicar_control(model, data, indices, angulos)
            
            # Avanzar simulación
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)
    
    print("✓ Saludo completado")


# ===========================
# FUNCIÓN PRINCIPAL
# ===========================

def main():
    """
    Función principal que ejecuta toda la secuencia del experimento.
    """
    print("\n" + "="*70)
    print("EXPERIMENTO 01: SALUDO CON UNITREE G1")
    print("="*70 + "\n")
    
    # Paso 1: Cargar el modelo
    print("📂 Cargando modelo...")
    try:
        model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        data = mujoco.MjData(model)
        print(f"✓ Modelo cargado correctamente: {SCENE_PATH}\n")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Paso 2: Listar articulaciones y actuadores (para referencia)
    listar_articulaciones_y_actuadores(model)
    
    # Paso 3: Obtener índices de los actuadores del brazo derecho
    prefijos_brazo_derecho = [
        'right_shoulder_pitch',
        'right_shoulder_roll',
        'right_shoulder_yaw',
        'right_elbow',
        'right_wrist_roll',
        'right_wrist_pitch',
        'right_wrist_yaw'
    ]
    
    indices = obtener_indices_actuadores(model, prefijos_brazo_derecho)
    
    print("🎯 Actuadores del brazo derecho detectados:")
    for nombre, indice in indices.items():
        print(f"   {nombre:30s} -> Índice {indice}")
    print()
    
    # Verificar que se encontraron todos los actuadores necesarios
    if len(indices) < len(prefijos_brazo_derecho):
        print("⚠️  Advertencia: No se encontraron todos los actuadores esperados")
    
    # Paso 4: Abrir el visor
    print("🖥️  Abriendo visor de MuJoCo...")
    print("   (Puedes rotar la cámara con el ratón durante la simulación)\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Paso 5: Estabilizar la simulación
        estabilizar_simulacion(model, data, viewer, STABILIZATION_TIME)
        
        if not viewer.is_running():
            return
        
        # Paso 6: Levantar el brazo
        mover_brazo(
            model, data, viewer, indices,
            angulos_inicio=ARM_INITIAL_ANGLES,
            angulos_fin=ARM_RAISED_ANGLES,
            duracion=ARM_RAISE_TIME,
            descripcion="Levantando brazo derecho"
        )
        
        if not viewer.is_running():
            return
        
        # Paso 7: Ejecutar el saludo
        ejecutar_saludo(
            model, data, viewer, indices,
            num_ciclos=NUMBER_OF_WAVES,
            duracion_ciclo=WAVE_CYCLE_TIME,
            amplitud=WAVE_AMPLITUDE
        )
        
        if not viewer.is_running():
            return
        
        # Paso 8: Bajar el brazo
        mover_brazo(
            model, data, viewer, indices,
            angulos_inicio=ARM_RAISED_ANGLES,
            angulos_fin=ARM_INITIAL_ANGLES,
            duracion=ARM_LOWER_TIME,
            descripcion="Bajando brazo derecho"
        )
        
        if not viewer.is_running():
            return
        
        # Paso 9: Pausa final
        print(f"⏸️  Pausa final ({FINAL_PAUSE_TIME:.1f}s)...")
        estabilizar_simulacion(model, data, viewer, FINAL_PAUSE_TIME)
        
        print("\n" + "="*70)
        print("✅ EXPERIMENTO COMPLETADO CON ÉXITO")
        print("="*70 + "\n")
        
        # Mantener el visor abierto hasta que el usuario lo cierre
        print("💡 El visor permanecerá abierto.")
        print("   Cierra la ventana para finalizar el programa.\n")
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1.0 / SIMULATION_FREQUENCY)


if __name__ == "__main__":
    main()
