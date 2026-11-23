"""Convierte un escaneo (gltf u obj, una o varias texturas) a partes OBJ + PNG para MuJoCo.
Uso: python convertir.py <archivo> <carpeta_salida> [--yup]"""
import os, sys
import numpy as np
import trimesh
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
src, out = sys.argv[1], sys.argv[2]
yup = "--yup" in sys.argv
os.makedirs(out, exist_ok=True)
s = trimesh.load(src, process=False)
if isinstance(s, trimesh.Trimesh):
    s = trimesh.Scene(s)
R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]) if yup else np.eye(4)
todos = []
n = 0
for nodo in s.graph.nodes_geometry:
    T, gname = s.graph[nodo]
    g = s.geometry[gname].copy()
    g.apply_transform(R @ T)
    mat = g.visual.material
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if img is None or g.visual.uv is None:
        print("sin textura:", gname)
        continue
    img.convert("RGB").save(os.path.join(out, f"tex_{n}.png"))
    with open(os.path.join(out, f"parte_{n}.obj"), "w") as f:
        f.writelines(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n" for v in g.vertices)
        f.writelines(f"vt {t[0]:.6f} {t[1]:.6f}\n" for t in g.visual.uv)
        f.writelines(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in g.faces + 1)
    todos.append(g.vertices)
    print(n, gname, len(g.vertices), len(g.faces), img.size)
    n += 1
v = np.concatenate(todos)
print("partes", n, "min", v.min(0).round(2), "max", v.max(0).round(2))
for ax in range(3):
    h, e = np.histogram(v[:, ax], bins=12)
    print(ax, " ".join(f"{e[i]:.1f}:{h[i]}" for i in range(12)))
