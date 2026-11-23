"""Arma un escenario de navegacion (escenarios/escaneo_<nombre>/escenario.xml) a partir de un escaneo de escaneos/<carpeta>/mj.

El escaneo es solo visual (no choca), pero el LiDAR y la camara lo ven. El robot camina sobre el piso plano de scene.xml, asi que el
escaneo se baja para que su piso quede en z = 0 y se corre para que el punto mas despejado de la sala quede en el origen, que es donde
arranca el robot.

Uso: python escaneos/prueba_integral/crear_escena.py --carpeta lost-places-hospital --nombre hospital --escala 0.25 --partes 4
"""
import argparse
import math
import os

import mujoco
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--carpeta", required=True)
ap.add_argument("--nombre", required=True)
ap.add_argument("--escala", type=float, required=True)
ap.add_argument("--partes", type=int, required=True)
ap.add_argument("--paso", type=float, default=0.25)
ap.add_argument("--textura", type=int, help="lado maximo de las texturas en pixeles (se guardan achicadas en escaneos/<carpeta>/mj_<lado>), para ahorrar memoria")
ap.add_argument("--piso", type=float, help="altura del piso del escaneo (m); sin esto, la mediana de las alturas del piso medidas")
a = ap.parse_args()

mj = os.path.join(RAIZ, "escaneos", a.carpeta, "mj")


def modelo(prefijo, cuerpo="", tex_prefijo=None):
    tex_prefijo = prefijo if tex_prefijo is None else tex_prefijo
    assets = "".join(f'<mesh name="esc_p{i}" file="{prefijo}parte_{i}.obj" scale="{a.escala} {a.escala} {a.escala}"/>'
                     f'<texture name="esc_t{i}" type="2d" file="{tex_prefijo}tex_{i}.png"/>'
                     f'<material name="esc_m{i}" texture="esc_t{i}" specular="0" shininess="0"/>' for i in range(a.partes))
    geoms = "".join(f'<geom name="esc_g{i}" type="mesh" mesh="esc_p{i}" material="esc_m{i}" group="0" contype="0" conaffinity="0"/>'
                    for i in range(a.partes))
    return assets, f"<body name=\"escaneo\" {cuerpo}>{geoms}</body>"


assets, cuerpo = modelo("")
m = mujoco.MjModel.from_xml_string(f'<mujoco><compiler meshdir="{mj}" texturedir="{mj}"/><asset>{assets}</asset>'
                                   f'<worldbody>{cuerpo}</worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
gid = np.array([-1], dtype=np.int32)


def rayo(p, v):
    v = np.asarray(v, float)
    return mujoco.mj_ray(m, d, np.asarray(p, float), v / np.linalg.norm(v), None, 1, -1, gid)


pts = np.concatenate([m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
                      @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in range(m.ngeom)])
lo, hi = pts.min(0), pts.max(0)
zp = np.percentile(pts[:, 2], 5)
candidatos = []
for x in np.arange(lo[0] + 0.5, hi[0] - 0.5, a.paso):
    for y in np.arange(lo[1] + 0.5, hi[1] - 0.5, a.paso):
        dpiso = rayo([x, y, zp + 1.0], [0, 0, -1])
        if dpiso < 0 or dpiso > 1.5:
            continue
        piso = zp + 1.0 - dpiso
        c = np.array([x, y, piso + 1.0])
        dist = np.array([rayo(c, [math.cos(t), math.sin(t), 0]) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)])
        if (dist < 0).any():
            continue
        candidatos.append((float(dist.min()), x, y, piso))
if not candidatos:
    raise SystemExit("no hay ningun punto cerrado por paredes en todas las direcciones")
pisos = np.array([q[3] for q in candidatos])
z_piso = float(np.median(pisos)) if a.piso is None else a.piso
libre, x0, y0, _ = max(q for q in candidatos if abs(q[3] - z_piso) < 0.08)
print(f"limites {lo.round(2)} {hi.round(2)}; piso {z_piso:.2f} m (desvio {pisos.std():.2f}); arranque ({x0:.2f}, {y0:.2f}) con {libre:.2f} m libres")

prefijo = f"../../escaneos/{a.carpeta}/mj/"
if a.textura:
    from PIL import Image
    chico = os.path.join(RAIZ, "escaneos", a.carpeta, f"mj_{a.textura}")
    os.makedirs(chico, exist_ok=True)
    for i in range(a.partes):
        im = Image.open(os.path.join(mj, f"tex_{i}.png"))
        im.thumbnail((a.textura, a.textura))
        im.save(os.path.join(chico, f"tex_{i}.png"))
    tex_prefijo = f"../../escaneos/{a.carpeta}/mj_{a.textura}/"
else:
    tex_prefijo = prefijo
assets, cuerpo = modelo(prefijo, f'pos="{-x0:.4f} {-y0:.4f} {-z_piso:.4f}"', tex_prefijo)
salida = os.path.join(RAIZ, "escenarios", f"escaneo_{a.nombre}")
os.makedirs(salida, exist_ok=True)
with open(os.path.join(salida, "escenario.xml"), "w", encoding="utf-8") as f:
    f.write(f"""<mujocoinclude>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
  </visual>
  <asset>
    {assets}
  </asset>
  <worldbody>
    {cuerpo}
  </worldbody>
</mujocoinclude>
""")
print("escrito", os.path.join(salida, "escenario.xml"))
