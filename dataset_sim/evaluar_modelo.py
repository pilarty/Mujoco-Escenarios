"""
Mide como responde un modelo de deteccion sobre las capturas simuladas.

Usa el mismo preprocesamiento y la misma calibracion que el backend
(redimensionar a 224x224, temperatura y umbral del archivo de calibracion).

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/evaluar_modelo.py --capturas dataset_sim/linea_base
"""
import argparse
import csv
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(os.path.dirname(RAIZ), "humanoid-damage-cnn", "models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capturas", required=True)
    ap.add_argument("--modelo", default=os.path.join(MODELOS, "modelo_detection3.keras"))
    ap.add_argument("--calibracion", default=os.path.join(MODELOS, "modelo_detection3_calibracion.json"))
    args = ap.parse_args()

    with open(args.calibracion, encoding="utf-8") as f:
        calib = json.load(f)
    T, umbral = float(calib.get("temperatura", 1.0)), float(calib.get("umbral_dano", 0.5))
    modelo = tf.keras.models.load_model(args.modelo)

    with open(os.path.join(args.capturas, "capturas.csv"), encoding="utf-8") as f:
        filas = [r for r in csv.DictReader(f) if r["clase"] != "descartar"]
    lote = np.stack([np.array(Image.open(os.path.join(args.capturas, r["archivo"])).convert("RGB").resize((224, 224)))
                     for r in filas]).astype("float32")
    p_sin = np.clip(modelo.predict(lote, batch_size=32, verbose=0).reshape(-1), 1e-6, 1 - 1e-6)
    p_dano = 1 / (1 + np.exp(-np.log((1 - p_sin) / p_sin) / T))
    dice_dano = p_dano >= umbral

    real = np.array([r["clase"] == "damaged" for r in filas])
    print(f"modelo: {os.path.basename(args.modelo)} | temperatura {T} | umbral de daño {umbral}")
    print(f"fotos sin daño: {(~real).sum()} | marcadas CON daño (falsos positivos): "
          f"{(dice_dano & ~real).sum()} ({100 * (dice_dano & ~real).sum() / max(1, (~real).sum()):.1f} %)")
    print(f"fotos con daño: {real.sum()} | detectadas: {(dice_dano & real).sum()} "
          f"({100 * (dice_dano & real).sum() / max(1, real.sum()):.1f} %)")
    print(f"aciertos totales: {100 * (dice_dano == real).mean():.1f} %")
    print(f"P(daño) mediana en fotos sin daño: {np.median(p_dano[~real]):.2f} | en fotos con daño: {np.median(p_dano[real]):.2f}")

    with open(os.path.join(args.capturas, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "clase", "p_dano", "dice_dano"])
        for r, p, d in zip(filas, p_dano, dice_dano):
            w.writerow([r["archivo"], r["clase"], round(float(p), 4), int(d)])


if __name__ == "__main__":
    main()
