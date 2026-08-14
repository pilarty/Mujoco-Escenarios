"""
Extrae las dimensiones reales de la habitación a partir del mesh (.obj) que
usa escenario.xml, y genera reporte/room_dimensions.json para que la
interfaz web pueda dibujar un plano a escala sin que nadie suba una foto.

A diferencia de un escenario armado con paredes individuales (geoms box),
acá toda la habitación es UN solo mesh 3D. Entonces el script:
  1. Lee escenario.xml y encuentra qué mesh usa, con qué escala, y en qué
     posición/rotación está colocado el body que lo contiene.
  2. Lee los vértices del archivo .obj (líneas "v x y z").
  3. Les aplica la escala y la transformación (pos/euler) del body, para
     llevarlos a coordenadas del mundo (las mismas que usa MuJoCo/el robot).
  4. Calcula el contorno (envolvente convexa) en x,y de esos vértices, que
     se usa como el perímetro del plano.

Uso:
    python extraer_dimensiones.py
"""

import os
import math
import json
import xml.etree.ElementTree as ET

# Debe coincidir con el ESCENARIO usado en el script de simulación
ESCENARIO = "habitacion_daniada"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ESCENARIO_DIR = os.path.join(PROJECT_ROOT, "escenarios", ESCENARIO)
ESCENARIO_XML = os.path.join(ESCENARIO_DIR, "escenario.xml")

REPORTE_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "reporte")
OUT_PATH = os.path.join(REPORTE_DIR, "room_dimensions.json")


def leer_vector(texto, default):
    if texto is None:
        return default
    return [float(v) for v in texto.split()]


def euler_a_matriz(ex, ey, ez):
    """Convierte ángulos de Euler en grados (convención por defecto de
    MuJoCo: rotación extrínseca en el orden x, y, z) a una matriz 3x3."""
    ex, ey, ez = math.radians(ex), math.radians(ey), math.radians(ez)
    cx, sx = math.cos(ex), math.sin(ex)
    cy, sy = math.cos(ey), math.sin(ey)
    cz, sz = math.cos(ez), math.sin(ez)

    Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]

    def mult(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return mult(Rz, mult(Ry, Rx))


def aplicar_transform(v, escala, pos, R):
    x = v[0] * escala[0]
    y = v[1] * escala[1]
    z = v[2] * escala[2]
    rx = R[0][0] * x + R[0][1] * y + R[0][2] * z
    ry = R[1][0] * x + R[1][1] * y + R[1][2] * z
    rz = R[2][0] * x + R[2][1] * y + R[2][2] * z
    return (rx + pos[0], ry + pos[1], rz + pos[2])


def leer_vertices_obj(path):
    vertices = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for linea in f:
            if linea.startswith("v "):
                partes = linea.split()
                x, y, z = float(partes[1]), float(partes[2]), float(partes[3])
                vertices.append((x, y, z))
    return vertices


def envolvente_convexa(puntos):
    """Envolvente convexa 2D (monotone chain), sin dependencias externas."""
    pts = sorted(set(puntos))
    if len(pts) <= 2:
        return pts

    def cruz(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    inferior = []
    for p in pts:
        while len(inferior) >= 2 and cruz(inferior[-2], inferior[-1], p) <= 0:
            inferior.pop()
        inferior.append(p)

    superior = []
    for p in reversed(pts):
        while len(superior) >= 2 and cruz(superior[-2], superior[-1], p) <= 0:
            superior.pop()
        superior.append(p)

    return inferior[:-1] + superior[:-1]


if __name__ == "__main__":
    if not os.path.exists(ESCENARIO_XML):
        raise FileNotFoundError(f"No se encontró {ESCENARIO_XML}")

    root = ET.parse(ESCENARIO_XML).getroot()

    # Mapea nombre de mesh -> (archivo, escala)
    meshes = {}
    for mesh in root.iter("mesh"):
        nombre = mesh.get("name")
        archivo = mesh.get("file")
        escala = leer_vector(mesh.get("scale"), [1.0, 1.0, 1.0])
        meshes[nombre] = {"file": archivo, "scale": escala}

    if not meshes:
        raise RuntimeError("No se encontró ningún <mesh> en escenario.xml")

    todos_los_puntos = []

    for body in root.iter("body"):
        pos = leer_vector(body.get("pos"), [0.0, 0.0, 0.0])
        euler = leer_vector(body.get("euler"), [0.0, 0.0, 0.0])
        R = euler_a_matriz(*euler)

        for geom in body.findall("geom"):
            mesh_name = geom.get("mesh")
            if not mesh_name or mesh_name not in meshes:
                continue

            info = meshes[mesh_name]
            # El path del XML es relativo a otra carpeta de compilación
            # (unitree_g1/...); acá lo resolvemos directo contra la carpeta
            # meshes/ del propio escenario, usando el nombre de archivo.
            nombre_archivo = os.path.basename(info["file"])
            obj_path = os.path.join(ESCENARIO_DIR, "meshes", nombre_archivo)

            if not os.path.exists(obj_path):
                print(f"Aviso: no se encontró el mesh en {obj_path}, se omite.")
                continue

            vertices = leer_vertices_obj(obj_path)
            for v in vertices:
                wx, wy, wz = aplicar_transform(v, info["scale"], pos, R)
                todos_los_puntos.append((wx, wy))

    if not todos_los_puntos:
        print("No se pudieron extraer vértices de ningún mesh. Revisá escenario.xml a mano.")
    else:
        xs = [p[0] for p in todos_los_puntos]
        ys = [p[1] for p in todos_los_puntos]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        contorno = envolvente_convexa(todos_los_puntos)

        room = {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width": x_max - x_min,
            "depth": y_max - y_min,
            "floor_polygon": [[round(x, 4), round(y, 4)] for x, y in contorno],
        }

        os.makedirs(REPORTE_DIR, exist_ok=True)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(room, f, ensure_ascii=False, indent=2)

        print(f"Habitación: {room['width']:.2f} m x {room['depth']:.2f} m")
        print(f"{len(todos_los_puntos)} vértices leídos, {len(contorno)} puntos en el contorno")
        print(f"Guardado en {OUT_PATH}")