# Comandos de entrenamiento y evaluacion - Unitree G1

Esta guia resume los comandos vigentes para entrenar y evaluar locomocion con LocoMuJoCo usando datasets locales `.npz`.

## 1. Crear entorno virtual desde cero

Ejecutar desde la raiz del repositorio clonado.

```powershell
python --version
python -m venv venv
```

Activar el entorno:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Actualizar herramientas base:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

Instalar dependencias completas del repositorio:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements\Base.txt
```

`requirements\Base.txt` incluye `requirements\locomujoco.txt`, que instala el framework oficial:

```powershell
git+https://github.com/robfiras/loco-mujoco.git
```

No hace falta clonar manualmente el repositorio oficial de LocoMuJoCo dentro del proyecto. `pip` lo descarga e instala como paquete Python.

Si se quiere reinstalar solo LocoMuJoCo sin reinstalar todo:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements\locomujoco.txt
```

Si la maquina tiene GPU NVIDIA y se quiere intentar JAX con CUDA, revisar primero la version de CUDA disponible. En caso contrario, usar JAX CPU, que es suficiente para validar el flujo aunque sea mas lento.

## 2. Verificar instalacion

Comprobar imports principales:

```powershell
.\venv\Scripts\python.exe -c "import mujoco, jax, loco_mujoco; from loco_mujoco import RLFactory, ImitationFactory; print('MuJoCo', mujoco.__version__); print('JAX', jax.__version__); print('LocoMuJoCo', loco_mujoco.__version__)"
```

Validar entorno G1 sin abrir viewer:

```powershell
.\venv\Scripts\python.exe tests\diagnose_g1_official_flow.py
```

Verificar que los datasets locales existen:

```powershell
Get-ChildItem "training\Movements npz\walk*.npz"
```

## 3. Activar entorno en sesiones futuras

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Tambien se puede ejecutar todo sin activar el entorno usando:

```powershell
.\venv\Scripts\python.exe <script>
```

## 4. Validar replay del dataset

Usar esto para confirmar que MuJoCo visualiza correctamente la trayectoria de referencia:

```powershell
.\venv\Scripts\python.exe tests\replay_g1_dataset.py --local-npz "training\Movements npz\walk1_subject1.npz"
```

Sin viewer:

```powershell
.\venv\Scripts\python.exe tests\replay_g1_dataset.py --local-npz "training\Movements npz\walk1_subject1.npz" --no-render --steps 1
```

## 5. Smoke test de entrenamiento

Script dedicado:

```powershell
.\venv\Scripts\python.exe training\train_g1_smoke.py
```

Equivalente manual:

```powershell
.\venv\Scripts\python.exe training\train_g1_imitation.py --local-npz "training\Movements npz\walk1_subject1.npz" --timesteps 10000 --num-envs 4 --num-steps 10
```

Objetivo:

- validar que carga el `.npz` local;
- crear ambiente con `ImitationFactory`;
- compilar PPOJax;
- entrenar una corrida corta;
- guardar `PPOJax_saved.pkl`.

No se espera una locomocion buena con 10k timesteps.

## 6. Evaluar politica entrenada

Sin viewer, recomendado en computadora lenta:

```powershell
.\venv\Scripts\python.exe tests\evaluate_g1_policy.py --no-render --steps 200
```

Visual liviano:

```powershell
.\venv\Scripts\python.exe tests\evaluate_g1_policy.py --steps 100 --sleep 0.05 --render-every 5
```

Visual mas completo, usar solo si la computadora lo soporta:

```powershell
.\venv\Scripts\python.exe tests\evaluate_g1_policy.py --steps 500 --sleep 0.03 --render-every 2
```

## 7. Entrenamiento largo con multiples caminatas

Script dedicado para pruebas largas:

```powershell
.\venv\Scripts\python.exe training\train_g1_long_walks.py
```

Por defecto usa estas trayectorias:

```text
walk1_subject1.npz
walk1_subject2.npz
walk1_subject5.npz
walk2_subject1.npz
walk2_subject3.npz
walk2_subject4.npz
walk3_subject1.npz
walk4_subject1.npz
```

Parametros por defecto:

```text
timesteps: 500000
num-envs: 16
num-steps: 50
```

## 8. Entrenamiento largo con parametros personalizados

100k, primera corrida corta real:

```powershell
.\venv\Scripts\python.exe training\train_g1_long_walks.py --timesteps 100000 --num-envs 8 --num-steps 25
```

500k, corrida intermedia:

```powershell
.\venv\Scripts\python.exe training\train_g1_long_walks.py --timesteps 500000 --num-envs 16 --num-steps 50
```

1M, corrida mas seria:

```powershell
.\venv\Scripts\python.exe training\train_g1_long_walks.py --timesteps 1000000 --num-envs 32 --num-steps 50
```

2.5M, solo si las anteriores mejoran:

```powershell
.\venv\Scripts\python.exe training\train_g1_long_walks.py --timesteps 2500000 --num-envs 64 --num-steps 100
```

## 9. Usar todos los `walk*.npz`

Si se quiere entrenar con todas las caminatas disponibles:

```powershell
.\venv\Scripts\python.exe training\train_g1_long_walks.py --all-walks --timesteps 500000 --num-envs 16 --num-steps 50
```

## 10. Orden recomendado

```text
1. Replay del .npz
2. Smoke test: train_g1_smoke.py
3. Evaluacion sin render
4. Entrenamiento 100k con train_g1_long_walks.py
5. Evaluacion sin render
6. Evaluacion visual si la maquina lo permite
7. Entrenamiento 500k
8. Evaluacion
9. Escalar a 1M o mas solo si hay mejora
```

## 11. Replicar en una computadora nueva

En una computadora nueva, el orden recomendado es:

```text
1. Clonar el repositorio.
2. Crear venv.
3. Instalar requirements/Base.txt, que tambien instala LocoMuJoCo.
4. Verificar que LocoMuJoCo importa correctamente.
5. Verificar imports.
6. Verificar datasets locales.
7. Ejecutar replay sin viewer.
8. Ejecutar replay visual.
9. Ejecutar smoke test.
10. Evaluar sin render.
11. Recién entonces entrenar largo.
```

Comandos compactos:

```powershell
git clone <URL_DEL_REPO>
cd Mujoco-Escenarios
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\venv\Scripts\python.exe -m pip install -r requirements\Base.txt
.\venv\Scripts\python.exe tests\diagnose_g1_official_flow.py
.\venv\Scripts\python.exe tests\replay_g1_dataset.py --local-npz "training\Movements npz\walk1_subject1.npz" --no-render --steps 1
```

## 12. Artefactos que no se versionan

Estos archivos/carpetas se regeneran y no deberian copiarse salvo que se quiera conservar una politica especifica:

```text
venv/
training/imitation/
LocoMuJoCo_recordings/
__pycache__/
```

Si se quiere mover una politica entrenada ya existente, copiar manualmente:

```text
training/imitation/local/.../PPOJax_saved.pkl
```

## 13. Notas

- `training/imitation/` contiene politicas generadas y esta ignorado por Git.
- `training/Movements npz/` contiene datasets locales versionados.
- HuggingFace esta bloqueado en el entorno corporativo; usar siempre rutas locales.
- La evaluacion visual puede ir lenta en computadoras sin buena capacidad grafica.
- LocoMuJoCo se instala como paquete Python desde `requirements\locomujoco.txt`; no se debe versionar ni copiar la carpeta temporal generada por `pip`.
