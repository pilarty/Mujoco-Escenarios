"""Vistas a la altura de la camara del G1 (1,2 m, fovy 90) dentro de un escaneo OBJ."""
import argparse, csv, math, os
import mujoco
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--carpeta", required=True)
ap.add_argument("--obj", default="modelnew.obj")
ap.add_argument("--tex", default="modelnew.png")
ap.add_argument("--escala", type=float, default=0.37)
ap.add_argument("--salida", required=True)
ap.add_argument("--paso", type=float, default=1.0)
ap.add_argument("--luz", type=float, default=0.55)
ap.add_argument("--partes", type=int, default=0)
ap.add_argument("--abierto", type=int, default=3)
a = ap.parse_args()

if a.partes:
    pares = [(f"parte_{i}.obj", f"tex_{i}.png") for i in range(a.partes)]
else:
    pares = [(a.obj, a.tex)]
ASSETS = "".join(f'<mesh name="p{i}" file="{o}" scale="{a.escala} {a.escala} {a.escala}"/><texture name="t{i}" type="2d" file="{t}"/><material name="m{i}" texture="t{i}" specular="0" shininess="0"/>' for i, (o, t) in enumerate(pares))
GEOMS = "".join(f'<geom type="mesh" mesh="p{i}" material="m{i}"/>' for i in range(len(pares)))
xml = f"""
<mujoco>
  <compiler meshdir="{a.carpeta}" texturedir="{a.carpeta}"/>
  <visual><global offwidth="640" offheight="480" fovy="90"/><quality shadowsize="0"/>
    <headlight ambient="{a.luz} {a.luz} {a.luz}" diffuse="0.35 0.35 0.35" specular="0 0 0"/></visual>
  <asset>
    {ASSETS}
  </asset>
  <worldbody>{GEOMS}</worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
gid = np.array([-1], dtype=np.int32)


def rayo(p, v):
    v = np.asarray(v, float)
    return mujoco.mj_ray(m, d, np.asarray(p, float), v / np.linalg.norm(v), None, 1, -1, gid)


pts = []
for g in range(m.ngeom):
    mid = m.geom_dataid[g]
    a0, n0 = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    pts.append(m.mesh_vert[a0:a0 + n0] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
pts = np.concatenate(pts)
lo, hi = pts.min(0), pts.max(0)
zp = np.percentile(pts[:, 2], 5)
print("limites", lo.round(2), hi.round(2), "piso aprox", round(zp, 2))

os.makedirs(os.path.join(a.salida, "frames"), exist_ok=True)
r = mujoco.Renderer(m, 480, 640)
cam = mujoco.MjvCamera()
filas = []
n = 0
for x in np.arange(lo[0] + 0.5, hi[0], a.paso):
    for y in np.arange(lo[1] + 0.5, hi[1], a.paso):
        z0 = zp + 1.0
        dpiso = rayo([x, y, z0], [0, 0, -1])
        if dpiso < 0 or dpiso > 1.5:
            continue
        piso = z0 - dpiso
        c = np.array([x, y, piso + 1.2])
        dist = []
        for k in range(16):
            t = 2 * math.pi * k / 16
            dist.append(rayo(c, [math.cos(t), math.sin(t), 0]))
        dist = np.array(dist)
        if (dist < 0).sum() > a.abierto or (dist[dist > 0] < 0.5).any():
            continue
        for k in range(0, 16, 2):
            t = 2 * math.pi * k / 16
            dv = np.array([math.cos(t), math.sin(t), math.tan(math.radians(-10))])
            dp = rayo(c, dv)
            if dp < 0:
                continue
            cam.lookat[:] = c + dv / np.linalg.norm(dv) * 0.5
            cam.distance = 0.5
            cam.azimuth = math.degrees(t)
            cam.elevation = -10
            r.update_scene(d, camera=cam)
            img = r.render()
            nombre = f"{n:04d}.png"
            Image.fromarray(img).save(os.path.join(a.salida, "frames", nombre))
            filas.append(dict(archivo=f"frames/{nombre}", clase="damaged", x=round(x, 2), y=round(y, 2),
                              yaw=round(t, 3), dist_pared=round(float(dp), 2), accion="vista", sala="", metros=0))
            n += 1
with open(os.path.join(a.salida, "capturas.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
    w.writeheader()
    w.writerows(filas)
print(n, "vistas")
