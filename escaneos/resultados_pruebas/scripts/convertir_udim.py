"""OBJ con varios materiales (UDIM de RealityCapture) -> partes OBJ + PNG.
Uso: python convertir_udim.py <obj> <carpeta_texturas> <salida> [--yup]"""
import os, re, sys
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
obj, texdir, out = sys.argv[1:4]
yup = "--yup" in sys.argv
os.makedirs(out, exist_ok=True)
V, VT, grupos, actual = [], [], {}, None
for l in open(obj):
    if l.startswith("v "):
        V.append([float(x) for x in l.split()[1:4]])
    elif l.startswith("vt "):
        VT.append([float(x) for x in l.split()[1:3]])
    elif l.startswith("usemtl"):
        actual = l.split()[1].strip()
    elif l.startswith("f "):
        grupos.setdefault(actual, []).append([tuple(int(x) - 1 for x in p.split("/")[:2]) for p in l.split()[1:4]])
V, VT = np.array(V), np.array(VT)
if yup:
    V = V[:, [0, 2, 1]] * [1, -1, 1]
print("min", V.min(0).round(2), "max", V.max(0).round(2))
for ax in range(3):
    h, e = np.histogram(V[:, ax], bins=12)
    print(ax, " ".join(f"{e[i]:.1f}:{h[i]}" for i in range(12)))
n = 0
for mat, caras in grupos.items():
    m = re.search(r"_u(\d+)_v(\d+)$", mat or "")
    if not m:
        print("sin textura:", mat, len(caras))
        continue
    u0, v0 = int(m.group(1)) - 1, int(m.group(2)) - 1
    tex = [f for f in os.listdir(texdir) if re.search(rf"_u{u0 + 1}_v{v0 + 1}\.", f)]
    if not tex:
        print("falta textura", mat)
        continue
    _im = Image.open(os.path.join(texdir, tex[0])).convert("RGB"); _im.thumbnail((4096, 4096)); _im.save(os.path.join(out, f"tex_{n}.png"))
    with open(os.path.join(out, f"parte_{n}.obj"), "w") as f:
        k = 0
        for cara in caras:
            for vi, ti in cara:
                x, y, z = V[vi]
                f.write(f"v {x:.5f} {y:.5f} {z:.5f}\n")
                f.write(f"vt {VT[ti][0] - u0:.6f} {VT[ti][1] - v0:.6f}\n")
            f.write(f"f {k + 1}/{k + 1} {k + 2}/{k + 2} {k + 3}/{k + 3}\n")
            k += 3
    print(n, mat, len(caras))
    n += 1
print("partes", n)
