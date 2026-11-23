import collections, csv, json, os, sys
import numpy as np
import tensorflow as tf
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
M = r"C:\Users\pilar\Documents\humanoid-damage-cnn\models"
clases = json.load(open(os.path.join(M, "modelo_clasificacion3_clases.json"), encoding="utf-8"))
T = json.load(open(os.path.join(M, "modelo_clasificacion3_calibracion.json")))["temperatura"]
mod = tf.keras.models.load_model(os.path.join(M, "modelo_clasificacion3.keras"))
AQUI = os.path.dirname(os.path.abspath(__file__))

for escena in sys.argv[1:]:
    c = os.path.join(AQUI, escena)
    filas = [f for f in csv.DictReader(open(os.path.join(c, "predicciones_v3.csv"), encoding="utf-8")) if f["dice_dano"] == "1"]
    rutas = [os.path.join(c, f["archivo"]) for f in filas]
    top = []
    salida = []
    for i in range(0, len(rutas), 32):
        lote = np.stack([np.asarray(Image.open(r).convert("RGB").resize((224, 224)), dtype="float32") for r in rutas[i:i + 32]])
        p = mod.predict(lote, verbose=0)
        lg = np.log(np.clip(p, 1e-9, 1)) / T
        p = np.exp(lg - lg.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        for f, fila in zip(p, filas[i:i + 32]):
            k = int(f.argmax())
            top.append(clases[k])
            salida.append([fila["archivo"], clases[k], round(float(f[k]), 3)])
    with open(os.path.join(c, "clasificacion_v3.csv"), "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["archivo", "clase", "prob"]] + salida)
    print(f"\n{escena}: {len(top)} fotos marcadas con daño por la v3")
    for k, n in collections.Counter(top).most_common():
        print(f"  {k}: {n} ({100 * n / len(top):.0f} %)")
