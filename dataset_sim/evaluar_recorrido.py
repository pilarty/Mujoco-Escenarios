"""
Pasa el modelo de deteccion por todas las fotos de un recorrido del robot
(recorrido_robot.py) y arma un video con lo que dice el modelo en cada foto.

Usa el mismo preprocesamiento y la misma calibracion que el backend
(redimensionar a 224x224, temperatura y umbral del archivo de calibracion).

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/evaluar_recorrido.py --recorrido dataset_sim/recorrido_casa_blanca --version 3

Salida en la carpeta del recorrido:
    predicciones_v<N>.csv
    recorrido_v<N>.mp4   cada foto con un borde verde (sin daño) o rojo (daño)
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS = os.path.join(os.path.dirname(RAIZ), "humanoid-damage-cnn", "models")
VERDE, ROJO, GRIS = (60, 170, 80), (200, 50, 50), (150, 150, 150)


def puntuar(rutas, modelo, T):
    salida = []
    for i in range(0, len(rutas), 64):
        lote = np.stack([np.array(Image.open(r).convert("RGB").resize((224, 224))) for r in rutas[i:i + 64]])
        p = np.clip(modelo.predict(lote.astype("float32"), verbose=0).reshape(-1), 1e-6, 1 - 1e-6)
        salida.append(1 / (1 + np.exp(-np.log((1 - p) / p) / T)))
    return np.concatenate(salida)


def racha_mas_larga(marcas):
    mejor = actual = 0
    for m in marcas:
        actual = actual + 1 if m else 0
        mejor = max(mejor, actual)
    return mejor


def video(ruta, filas, rutas, p_dano, umbral, version):
    alto, ancho = 480, 640
    escritor = None
    for codec in ("avc1", "mp4v"):
        escritor = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*codec), 8, (ancho, alto + 60))
        if escritor.isOpened():
            break
    for f, r, p in zip(filas, rutas, p_dano):
        img = np.array(Image.open(r).convert("RGB"))
        dice = p >= umbral
        color = ROJO if dice else VERDE
        lienzo = np.full((alto + 60, ancho, 3), 255, np.uint8)
        lienzo[:alto] = img
        cv2.rectangle(lienzo, (0, 0), (ancho - 1, alto - 1), color, 10)
        lienzo[alto:] = color
        texto = f"modelo v{version}: {'DANO' if dice else 'sin dano'}  P(dano) {p:.2f}"
        real = {"damaged": "con dano", "undamaged": "sana", "descartar": "un poco de dano"}[f["clase"]]
        cv2.putText(lienzo, texto, (12, alto + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lienzo, f"pared en la foto: {real}   {f['sala']}   {float(f['metros']):.1f} m", (12, alto + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        escritor.write(cv2.cvtColor(lienzo, cv2.COLOR_RGB2BGR))
    escritor.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorrido", required=True)
    ap.add_argument("--version", default="3")
    ap.add_argument("--sin_video", action="store_true")
    args = ap.parse_args()

    ruta_modelo = os.path.join(MODELOS, f"modelo_detection{args.version}.keras")
    with open(os.path.join(MODELOS, f"modelo_detection{args.version}_calibracion.json"), encoding="utf-8") as f:
        calib = json.load(f)
    T, umbral = float(calib.get("temperatura", 1.0)), float(calib.get("umbral_dano", 0.5))
    modelo = tf.keras.models.load_model(ruta_modelo)

    with open(os.path.join(args.recorrido, "capturas.csv"), encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    rutas = [os.path.join(args.recorrido, f["archivo"]) for f in filas]
    p_dano = puntuar(rutas, modelo, T)
    dice = p_dano >= umbral
    clase = np.array([f["clase"] for f in filas])

    with open(os.path.join(args.recorrido, f"predicciones_v{args.version}.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "clase", "accion", "sala", "p_dano", "dice_dano"])
        for fi, p, d in zip(filas, p_dano, dice):
            w.writerow([fi["archivo"], fi["clase"], fi["accion"], fi["sala"], round(float(p), 4), int(d)])

    sanas, con_dano = clase == "undamaged", clase == "damaged"
    print(f"modelo v{args.version} | temperatura {T} | umbral de daño {umbral} | {len(filas)} fotos")
    print(f"fotos con la pared sana: {sanas.sum()} | marcadas CON daño: {(dice & sanas).sum()} "
          f"({100 * (dice & sanas).mean() / max(1e-9, sanas.mean()):.1f} %)")
    print(f"fotos con daño a la vista: {con_dano.sum()} | detectadas: {(dice & con_dano).sum()} "
          f"({100 * (dice & con_dano).mean() / max(1e-9, con_dano.mean()):.1f} %)")
    print(f"racha mas larga de fotos sanas seguidas marcadas con daño: {racha_mas_larga(dice & sanas)}")
    for accion in ("caminando", "girando", "vuelta", "mirando_pared"):
        m = sanas & (np.array([f["accion"] for f in filas]) == accion)
        if m.any():
            print(f"  {accion}: {m.sum()} fotos sanas, {100 * dice[m].mean():.1f} % marcadas con daño")
    if not args.sin_video:
        destino = os.path.join(args.recorrido, f"recorrido_v{args.version}.mp4")
        video(destino, filas, rutas, p_dano, umbral, args.version)
        print(f"video: {destino}")


if __name__ == "__main__":
    main()
