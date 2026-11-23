"""
Genera fotos de uno o varios escenarios desde la camara del robot para
entrenar y evaluar el detector de daños con imagenes simuladas. Sirve para
edificio_informe y para los edificios de escenarios/variantes.

Cada foto se etiqueta sola, usando lo que el simulador sabe de la escena:
  - "damaged":   se ve un daño de pared (grieta, humedad, desprendimiento)
  - "undamaged": no se ve ningun daño (pared sana, muebles, piso, cielo)
  - "descartar": se ven escombros, la viga caida o la brecha, o el daño de
                 pared ocupa muy poco; no se usan para entrenar

En cada escenario, cada variante cambia la luz y el color y el grano de las
texturas. La variante 0 es el escenario tal cual esta guardado.

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/generar_capturas.py --escenarios edificio_informe "variantes/edificio_v*"

Salida: dataset_sim/capturas/<escenario>/<clase>/*.png y dataset_sim/capturas/capturas.csv
"""
import argparse
import csv
import glob
import json
import math
import os

import mujoco
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMARA = "head_camera"
ALTO, ANCHO = 480, 640
ALTURA_BASE = 0.793
SEPARACION = 0.45                     # distancia minima a paredes y muebles
MIN_DANO = 0.02                       # fraccion de la foto con daño de pared para "damaged"
MAX_RESTO = 0.0005                    # fraccion maxima de daño para "undamaged"
TILE_REVOQUE = 2.0                    # igual que en generar_edificio.py
OBJETOS_DANO = ("escombro", "viga_caida", "panel_colgante")


def construir_modelo(escenario):
    unitree = os.path.join(RAIZ, "unitree_g1")
    rel = os.path.relpath(os.path.join(RAIZ, "escenarios", escenario), unitree).replace("\\", "/")
    xml = f'<mujoco><include file="scene.xml"/><include file="{rel}/escenario.xml"/></mujoco>'
    anterior = os.getcwd()
    try:
        os.chdir(unitree)
        modelo = mujoco.MjModel.from_xml_string(xml)
    finally:
        os.chdir(anterior)
    return modelo, mujoco.MjData(modelo)


def nombres_geoms(modelo):
    return [mujoco.mj_id2name(modelo, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(modelo.ngeom)]


def limites_y_brechas(modelo, datos, nombres, escenario):
    """Interior del edificio (segun sus paredes exteriores) y zonas de brecha del manifest."""
    ids = [i for i, n in enumerate(nombres) if n.startswith("col_ext_")]
    pos = np.array([datos.geom_xpos[i] for i in ids])
    limites = (pos[:, 0].min() + 0.4, pos[:, 0].max() - 0.4, pos[:, 1].min() + 0.4, pos[:, 1].max() - 0.4)
    with open(os.path.join(RAIZ, "escenarios", escenario, "damage_manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    brechas = []
    for d in manifest:
        if d.get("tipo") == "brecha_pared":
            cx, cy = d["centro"]
            m = d["ancho"] / 2 + 0.15
            brechas.append((cx - m, cx + m, cy - m, cy + m, d["alto"] + 0.6))
    return limites, brechas


def cargar_danos(modelo, datos, nombres, escenario):
    """Mascara de daño de cada pieza de pared, en metros sobre la pared.

    La textura de cada pieza es la pared sana con el daño pintado encima. Se
    compara contra la pared sana (la misma textura base, muestreada igual que
    en generar_edificio.py) y lo que difiere es daño."""
    carpeta = os.path.join(RAIZ, "escenarios", escenario)
    with open(os.path.join(carpeta, "damage_manifest.json"), encoding="utf-8") as f:
        manifest = [d for d in json.load(f) if "pieza" in d]
    base = np.array(Image.open(os.path.join(carpeta, "assets", "pared_revoque.png")).convert("RGB"), np.float32)
    S = base.shape[0]
    danos = {}
    for pieza in sorted({d["pieza"] for d in manifest}):
        atlas = np.array(Image.open(os.path.join(carpeta, "assets", f"pieza_{pieza}.png")).convert("RGB"), np.float32)
        Hp, Wp = atlas.shape[0] // 2, atlas.shape[1]
        for gid in [i for i, n in enumerate(nombres) if n.startswith(f"vis_pieza_{pieza}")]:
            centro = datos.geom_xpos[gid] + datos.geom_xmat[gid].reshape(3, 3) @ modelo.geom_aabb[gid][:3]
            medio = np.abs(datos.geom_xmat[gid].reshape(3, 3) @ modelo.geom_aabb[gid][3:])
            eje_normal = int(np.argmin(medio[:2]))          # la pared es fina en la direccion de su normal
            largo = 1 - eje_normal
            s0, s1 = centro[largo] - medio[largo], centro[largo] + medio[largo]
            z0, z1 = centro[2] - medio[2], centro[2] + medio[2]
            sm = s0 + (np.arange(Wp) + 0.5) / Wp * (s1 - s0)
            zm = z1 - (np.arange(Hp) + 0.5) / Hp * (z1 - z0)
            col = ((sm / TILE_REVOQUE) % 1.0 * S).astype(int) % S
            fil = ((1.0 - (zm / TILE_REVOQUE) % 1.0) * S).astype(int) % S
            sana = base[fil[:, None], col[None, :]]
            mascaras = {}
            for lado, trozo in ((1, atlas[:Hp]), (-1, atlas[Hp:])):
                m = np.abs(trozo - sana).max(axis=-1) > 14
                # solo cuenta dentro de los recuadros del manifest: afuera, las diferencias son redondeos del muestreo
                zona = np.zeros_like(m)
                for d in manifest:
                    if d["pieza"] == pieza and (1 if d["cara"].isupper() else -1) == lado:
                        zona |= ((np.abs(sm - d["centro_s"]) < d["ancho"] / 2 + 0.05)[None, :]
                                 & (np.abs(zm - d["centro_z"]) < d["alto"] / 2 + 0.05)[:, None])
                mascaras[lado] = maximum_filter(m & zona, size=13)   # engorda las grietas finas unos 3 cm
            tipos = {lado: sorted({d["tipo"] for d in manifest if d["pieza"] == pieza
                                   and (1 if d["cara"].isupper() else -1) == lado}) for lado in (1, -1)}
            danos[gid] = dict(eje_normal=eje_normal, pos=centro[eje_normal], s0=s0, s1=s1, z0=z0, z1=z1,
                              mascaras=mascaras, tipos=tipos)
    return danos


def dentro_de(poligonos, x, y):
    """True si (x, y) cae en alguno de los poligonos (salas de una casa de ProcTHOR)."""
    for pol in poligonos:
        dentro = False
        for i in range(len(pol)):
            (x1, y1), (x2, y2) = pol[i], pol[(i + 1) % len(pol)]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                dentro = not dentro
        if dentro:
            return True
    return False


def posiciones_libres(modelo, datos, rng, cuantas, limites, interior=None):
    """Puntos (x, y) dentro del edificio, lejos de paredes y muebles.
    Si el escenario trae interior.json (casas de ProcTHOR), el punto ademas tiene que caer en una sala."""
    grupos = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    angulos = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    geomid = np.zeros(1, dtype=np.int32)
    libres = []
    while len(libres) < cuantas:
        x = rng.uniform(limites[0], limites[1])
        y = rng.uniform(limites[2], limites[3])
        if interior and not dentro_de(interior, x, y):
            continue
        ok = True
        for z in (0.25, 0.9):
            for a in angulos:
                d = mujoco.mj_ray(modelo, datos, np.array([x, y, z]), np.array([math.cos(a), math.sin(a), 0.0]),
                                  grupos, 1, -1, geomid)
                if 0 <= d < SEPARACION:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            libres.append((x, y))
    return libres


def colocar_robot(modelo, datos, x, y, yaw, cabeceo, dz, postura):
    adr = modelo.joint("floating_base_joint").qposadr[0]
    datos.qpos[:] = postura
    datos.qpos[adr:adr + 3] = [x, y, ALTURA_BASE + dz]
    q_yaw = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    q_cab = np.array([math.cos(cabeceo / 2), 0, math.sin(cabeceo / 2), 0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_yaw, q_cab)
    datos.qpos[adr + 3:adr + 7] = q
    mujoco.mj_forward(modelo, datos)


def aplicar_variante(modelo, render, base, rng, numero):
    """Cambia luces y texturas. La variante 0 deja todo como en el escenario original."""
    modelo.light_diffuse[:] = base["luz"]
    modelo.vis.headlight.ambient[:] = base["ambiente"]
    modelo.vis.headlight.diffuse[:] = base["frontal"]
    modelo.tex_data[:] = base["tex"]
    descripcion = "original"
    if numero > 0:
        tono = rng.uniform(0.85, 1.12, 3)
        modelo.light_diffuse[:] = base["luz"] * rng.uniform(0.45, 1.45) * tono
        modelo.vis.headlight.ambient[:] = np.clip(base["ambiente"] * rng.uniform(0.5, 1.9), 0, 1)
        modelo.vis.headlight.diffuse[:] = np.clip(base["frontal"] * rng.uniform(0.5, 2.5), 0, 1)
        # paredes: un color de pintura comun; el resto de las texturas, un cambio leve
        pinturas = [(1.0, 1.0, 1.0), (0.97, 0.93, 0.84), (0.90, 0.90, 0.92), (0.93, 0.96, 0.90),
                    (0.98, 0.90, 0.82), (0.86, 0.91, 0.96), (0.80, 0.80, 0.78)]
        pintura = np.array(pinturas[rng.integers(len(pinturas))]) * rng.uniform(0.85, 1.05)
        grano = rng.uniform(0.6, 1.8)
        for t in range(modelo.ntex):
            nombre = mujoco.mj_id2name(modelo, mujoco.mjtObj.mjOBJ_TEXTURE, t) or ""
            if modelo.tex_type[t] != mujoco.mjtTexture.mjTEXTURE_2D or modelo.tex_nchannel[t] != 3:
                continue
            a, n = modelo.tex_adr[t], modelo.tex_height[t] * modelo.tex_width[t] * 3
            img = base["tex"][a:a + n].reshape(modelo.tex_height[t], modelo.tex_width[t], 3).astype(np.float32)
            if nombre.startswith("t_revoque") or nombre.startswith("t_pieza"):
                suave = gaussian_filter(img, sigma=(6, 6, 0))
                img = (suave + (img - suave) * grano) * pintura
            else:
                img = img * rng.uniform(0.8, 1.15) * rng.uniform(0.93, 1.07, 3)
            modelo.tex_data[a:a + n] = np.clip(img, 0, 255).astype(np.uint8).reshape(-1)
        descripcion = f"pintura {np.round(pintura, 2).tolist()}, grano {grano:.2f}"
    if render._gl_context is not None:
        render._gl_context.make_current()
    for t in range(modelo.ntex):
        mujoco.mjr_uploadTexture(modelo, render._mjr_context, t)
    return descripcion


def etiquetar(modelo, datos, render, cam_id, nombres, danos, brechas):
    """Devuelve (clase, fraccion de daño de pared, fraccion de otros daños, tipos vistos)."""
    render.enable_segmentation_rendering()
    render.update_scene(datos, camera=CAMARA)
    seg = render.render()[..., 0].copy()
    render.disable_segmentation_rendering()
    render.enable_depth_rendering()
    render.update_scene(datos, camera=CAMARA)
    prof = render.render().copy()
    render.disable_depth_rendering()

    f = (ALTO / 2) / math.tan(math.radians(modelo.cam_fovy[cam_id]) / 2)
    u, v = np.meshgrid(np.arange(ANCHO) + 0.5, np.arange(ALTO) + 0.5)
    rayos = np.stack([(u - ANCHO / 2) / f, -(v - ALTO / 2) / f, -np.ones_like(u)], axis=-1) * prof[..., None]
    R = datos.cam_xmat[cam_id].reshape(3, 3)
    pos_cam = datos.cam_xpos[cam_id]
    mundo = rayos @ R.T + pos_cam

    total = ALTO * ANCHO
    pared = np.zeros((ALTO, ANCHO), dtype=bool)
    tipos = set()
    for gid, d in danos.items():
        en_pieza = seg == gid
        if not en_pieza.any():
            continue
        n = d["eje_normal"]
        lado = 1 if pos_cam[n] > d["pos"] else -1
        m = d["mascaras"][lado]
        Hp, Wp = m.shape
        c = ((mundo[..., 1 - n] - d["s0"]) / (d["s1"] - d["s0"]) * Wp).astype(int).clip(0, Wp - 1)
        r = ((d["z1"] - mundo[..., 2]) / (d["z1"] - d["z0"]) * Hp).astype(int).clip(0, Hp - 1)
        dentro = en_pieza & m[r, c]
        if dentro.sum() > 0.003 * total:
            tipos.update(d["tipos"][lado])
        pared |= dentro

    ids_objetos = [i for i, n in enumerate(nombres) if n.startswith(OBJETOS_DANO)]
    objetos = np.isin(seg, ids_objetos)
    for x0, x1, y0, y1, zmax in brechas:
        objetos |= ((seg >= 0) & (mundo[..., 0] > x0) & (mundo[..., 0] < x1) & (mundo[..., 1] > y0)
                    & (mundo[..., 1] < y1) & (mundo[..., 2] < zmax) & (prof < 15))

    f_pared, f_obj = pared.sum() / total, objetos.sum() / total
    if f_pared >= MIN_DANO:
        clase = "damaged"
    elif f_pared <= MAX_RESTO and f_obj <= MAX_RESTO:
        clase = "undamaged"
    else:
        clase = "descartar"
    return clase, f_pared, f_obj, sorted(tipos)


def capturar_escenario(escenario, args, rng, filas):
    modelo, datos = construir_modelo(escenario)
    nombres = nombres_geoms(modelo)
    cam_id = mujoco.mj_name2id(modelo, mujoco.mjtObj.mjOBJ_CAMERA, CAMARA)
    etiqueta = escenario.split("/")[-1]

    with open(os.path.join(RAIZ, "g1_navigation_integration", "policy", "deploy.yaml"), encoding="utf-8") as f:
        articulaciones = np.array(yaml.safe_load(f)["default_joint_pos"], dtype=float)
    postura = datos.qpos.copy()
    adr = modelo.joint("floating_base_joint").qposadr[0]
    postura[adr + 7:adr + 7 + len(articulaciones)] = articulaciones

    # el robot se aparta mientras se buscan posiciones libres
    colocar_robot(modelo, datos, 0, 0, 0, 0, 100, postura)
    danos = cargar_danos(modelo, datos, nombres, escenario)
    limites, brechas = limites_y_brechas(modelo, datos, nombres, escenario)
    ruta_interior = os.path.join(RAIZ, "escenarios", escenario, "interior.json")
    interior = None
    if os.path.exists(ruta_interior):
        with open(ruta_interior, encoding="utf-8") as f:
            interior = json.load(f)

    render = mujoco.Renderer(modelo, ALTO, ANCHO)
    base = dict(luz=modelo.light_diffuse.copy(), ambiente=modelo.vis.headlight.ambient.copy(),
                frontal=modelo.vis.headlight.diffuse.copy(), tex=modelo.tex_data.copy())

    for clase in ("damaged", "undamaged", "descartar"):
        os.makedirs(os.path.join(args.salida, etiqueta, clase), exist_ok=True)
    # cerca de cada daño de pared se sacan fotos a proposito: al azar casi nunca se lo mira de frente
    objetivos = []
    for d in danos.values():
        for lado, m in d["mascaras"].items():
            filas_m, cols_m = np.nonzero(m)
            if len(cols_m) == 0:
                continue
            c = np.zeros(2)
            c[d["eje_normal"]] = d["pos"]
            c[1 - d["eje_normal"]] = d["s0"] + cols_m.mean() / m.shape[1] * (d["s1"] - d["s0"])
            objetivos.append((c, d["eje_normal"], lado))

    total = {"damaged": 0, "undamaged": 0, "descartar": 0}
    for v in range(args.variantes):
        colocar_robot(modelo, datos, 0, 0, 0, 0, 100, postura)
        libres = posiciones_libres(modelo, datos, rng, args.por_variante * 4, limites, interior)
        aplicar_variante(modelo, render, base, rng, v)
        hechas = {"damaged": 0, "undamaged": 0, "descartar": 0}
        cupo = {"damaged": args.por_variante // 2, "undamaged": args.por_variante - args.por_variante // 2,
                "descartar": args.por_variante // 10}
        for i, (x, y) in enumerate(libres):
            if hechas["damaged"] >= cupo["damaged"] and hechas["undamaged"] >= cupo["undamaged"]:
                break
            yaw = rng.uniform(-math.pi, math.pi)
            if hechas["damaged"] < cupo["damaged"] and i % 3 != 0 and objetivos:
                c, n, lado = objetivos[rng.integers(len(objetivos))]
                if np.hypot(c[0] - x, c[1] - y) < 5.0 and ((x, y)[n] - c[n]) * lado > 0:
                    yaw = math.atan2(c[1] - y, c[0] - x) + rng.normal(0, 0.25)
            cabeceo = rng.normal(0, math.radians(6))
            dz = rng.uniform(-0.05, 0.05)
            colocar_robot(modelo, datos, x, y, yaw, cabeceo, dz, postura)
            clase, f_pared, f_obj, tipos = etiquetar(modelo, datos, render, cam_id, nombres, danos, brechas)
            if hechas[clase] >= cupo[clase]:
                continue
            render.update_scene(datos, camera=CAMARA)
            nombre = f"v{v:02d}_{hechas[clase]:04d}.png"
            Image.fromarray(render.render()).save(os.path.join(args.salida, etiqueta, clase, nombre))
            hechas[clase] += 1
            total[clase] += 1
            filas.append(dict(archivo=f"{etiqueta}/{clase}/{nombre}", escenario=etiqueta, clase=clase, variante=v,
                              x=round(x, 3), y=round(y, 3), yaw=round(yaw, 3), dano_pared=round(f_pared, 4),
                              otros_danos=round(f_obj, 4), tipos="|".join(tipos)))
    render.close()
    print(f"{etiqueta}: {total}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escenarios", nargs="+", default=["edificio_informe"],
                    help='carpetas dentro de escenarios/; admite comodines, ej. "variantes/edificio_v*"')
    ap.add_argument("--variantes", type=int, default=4)
    ap.add_argument("--por_variante", type=int, default=40)
    ap.add_argument("--salida", default=os.path.join(RAIZ, "dataset_sim", "capturas"))
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    base_esc = os.path.join(RAIZ, "escenarios")
    escenarios = []
    for patron in args.escenarios:
        for e in sorted(glob.glob(os.path.join(base_esc, patron))):
            escenarios.append(os.path.relpath(e, base_esc).replace(os.sep, "/"))
    rng = np.random.default_rng(args.semilla)
    filas = []
    for escenario in escenarios:
        capturar_escenario(escenario, args, rng, filas)

    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "capturas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(f"listo: {len(filas)} fotos en {args.salida}")


if __name__ == "__main__":
    main()
