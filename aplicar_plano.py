"""Hace que el robot le mande al back el mapa de paredes ('paredes' de POST /room) y su posicion (POST /pose), que es lo que el front necesita para dibujar el plano y el recorrido.
Uso:  python aplicar_plano.py launch_g1_nav.py g1_nav_config.yaml
Hace una copia de cada archivo (.antes_de_plano) y, si algun ancla no esta en tu archivo, no toca nada y avisa cual."""
import py_compile
import re
import shutil
import sys

launch, yml = sys.argv[1], sys.argv[2]
s = open(launch, encoding="utf-8").read()
if "import plano" in s:
    sys.exit("[aplicar_plano] ese launch ya tiene el plano (import plano); no hago nada")
hechos = []


def una_vez(patron, nuevo, nombre, flags=re.M, opcional=False):
    """Reemplaza la unica coincidencia del patron (con una funcion para poder usar la sangria capturada)."""
    global s
    n = len(re.findall(patron, s, flags))
    if n == 0 and opcional:
        print(f"[aplicar_plano] aviso: no encontre el ancla de '{nombre}' (es solo estetico): lo salteo")
        return
    if n != 1:
        sys.exit(f"[aplicar_plano] no encontre (o hay {n}) el ancla de '{nombre}' en {launch}; no toque nada. Mandame esa parte del archivo.")
    s = re.sub(patron, nuevo, s, flags=flags)
    hechos.append(nombre)


una_vez(r"^import slam_vivo$", "import slam_vivo\nimport plano", "import plano")
una_vez(r'^([ \t]*)try:\n([ \t]*escenario_dir = os\.path\.join\(PROJECT_ROOT, "escenarios", SCENARIO\))',
        lambda m: f"{m.group(1)}room = None   # (si no se puede calcular el layout del escenario, plano.py arma los limites desde el mapa del SLAM)\n{m.group(1)}try:\n{m.group(2)}",
        "room = None antes de calcular el layout")
una_vez(r"^([ \t]*)(cob = cobertura\.cob_crear\()",
        lambda m: f"{m.group(1)}pl = plano.plano_crear(cfg_nav, room, BACKEND_ROOM_URL)   # mapa de paredes y posicion hacia el front (None si el yaml tiene 'plano_al_back: false')\n{m.group(1)}{m.group(2)}",
        "crear el estado del plano")
una_vez(r"^([ \t]*)step_count \+= 1\n([ \t]*if step_count % log_every == 0:)",
        lambda m: f"{m.group(1)}plano.plano_tick(pl, sl, data.time, base_pos, yaw, ex[\"terminada\"])   # posicion cada 0.5 s y paredes cada 'plano_cada_s', en un hilo\n{m.group(1)}step_count += 1\n{m.group(2)}",
        "mandar posicion y paredes durante la corrida")
una_vez(r"^([ \t]*)(if sl is not None and sl\[\"err\"\]:\n[ \t]*base_slam = )",
        lambda m: f"{m.group(1)}if pl is not None:\n{m.group(1)}    print(plano.plano_final(pl, sl))   # el mapa final, para que el front lo tenga completo\n{m.group(1)}{m.group(2)}",
        "mandar el mapa final")
una_vez(r'^([ \t]*)requests\.post\(BACKEND_ROOM_URL, json=room, timeout=5\)\n([ \t]*)print\(f"\[g1_nav\] Room layout enviado: (\{room\[\'width\'\]:\.2f\} m x \{room\[\'depth\'\]:\.2f\} m)"\)',
        lambda m: f"{m.group(1)}r_room = requests.post(BACKEND_ROOM_URL, json=room, timeout=5)\n{m.group(2)}print(f\"[g1_nav] Room layout enviado: {m.group(3)} (back: HTTP {{r_room.status_code}})\")",
        "mostrar el codigo HTTP del layout", opcional=True)
compile(s, launch, "exec")   # que al menos sea python valido antes de escribir nada
y = open(yml, encoding="utf-8").read()
if "plano_al_back" not in y:
    y = y.rstrip("\n") + """

# ---------------------------------------------------------------------------------------------------------------------------
# PLANO Y RECORRIDO PARA EL FRONT: el front dibuja el mapa con las celdas de pared que manda el robot y el camino con su posicion.
plano_al_back: true     # false: no se manda ni el mapa de paredes ni la posicion
plano_cada_s: 30        # cada cuantos segundos de simulacion se vuelve a mandar el mapa de paredes mientras crece
pose_al_back: true      # mandar la posicion del robot cada 0.5 s (el recorrido del front)
"""
shutil.copy(launch, launch + ".antes_de_plano")
shutil.copy(yml, yml + ".antes_de_plano")
open(launch, "w", encoding="utf-8").write(s)
open(yml, "w", encoding="utf-8").write(y)
py_compile.compile(launch, doraise=True)
print("[aplicar_plano] listo. Cambios en el launch:", "; ".join(hechos))
print("[aplicar_plano] el yaml quedo con el bloque 'plano_al_back' al final. Copias: *.antes_de_plano")
