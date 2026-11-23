"""
Elige fotos reales de daños del conjunto de clasificacion (bucket S3, formato
YOLO: cada foto trae recuadros con el tipo de daño) para pegarlas en las
paredes y pisos del simulador. Solo para probar en el simulador, no para entrenar.

Se queda con fotos que tienen un recuadro grande (al menos 12 % de la foto) de
grieta, humedad o revoque caido. Baja la foto y guarda el recuadro.

Uso (desde dataset_sim/danos_reales):
    python elegir_danos_reales.py
Salida: fotos/*.jpg y elegidas.csv (archivo, tipo, clase original, recuadro).
"""
import csv, os, random
from concurrent.futures import ThreadPoolExecutor
import boto3

BUCKET = "humanoid-damage-cnn-834257582282-us-east-1-an"
PREF = "processed/damage_classification/"
GRUPOS = {
    "grieta_estructural": {2, 4, 5, 13, 30, 36},
    "humedad": {31, 32, 33, 35, 37, 43},
    "desprendimiento_revoque": {0, 18, 21, 34},
}
CUPO = 120
AREA_MIN = 0.12

s3 = boto3.client("s3")
AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    claves = []
    for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=PREF + "labels/"):
        claves += [o["Key"] for o in pag.get("Contents", []) if o["Key"].endswith(".txt")]
    random.Random(0).shuffle(claves)
    print("etiquetas:", len(claves))
    grupo_de = {c: g for g, cs in GRUPOS.items() for c in cs}

    def leer(k):
        try:
            txt = s3.get_object(Bucket=BUCKET, Key=k)["Body"].read().decode()
        except Exception:
            return None
        cajas = []
        for l in txt.splitlines():
            p = l.split()
            if len(p) == 5:
                c, x, y, w, h = int(p[0]), *map(float, p[1:])
                cajas.append((c, x, y, w, h))
        buenas = [b for b in cajas if b[0] in grupo_de and b[3] * b[4] >= AREA_MIN and b[3] < 0.98 and b[4] < 0.98]
        if not buenas:
            return None
        b = max(buenas, key=lambda b: b[3] * b[4])
        return k, grupo_de[b[0]], b

    cuenta = {g: 0 for g in GRUPOS}
    elegidas = []
    with ThreadPoolExecutor(16) as ex:
        for i in range(0, len(claves), 800):
            for r in ex.map(leer, claves[i:i + 800]):
                if r and cuenta[r[1]] < CUPO:
                    cuenta[r[1]] += 1
                    elegidas.append(r)
            print(i + 800, cuenta, flush=True)
            if all(v >= CUPO for v in cuenta.values()) or i > 24000:
                break

    os.makedirs(os.path.join(AQUI, "fotos"), exist_ok=True)

    def bajar(r):
        k, g, b = r
        base = os.path.basename(k)[:-4]
        for ext in (".jpg", ".png", ".jpeg"):
            try:
                destino = os.path.join(AQUI, "fotos", base + ".jpg")
                obj = s3.get_object(Bucket=BUCKET, Key=PREF + "images/" + k.split("labels/")[1][:-4] + ext)
                with open(destino, "wb") as f:
                    f.write(obj["Body"].read())
                return dict(archivo=base + ".jpg", tipo=g, clase=b[0], x=b[1], y=b[2], w=b[3], h=b[4])
            except Exception:
                continue
        return None

    with ThreadPoolExecutor(16) as ex:
        filas = [f for f in ex.map(bajar, elegidas) if f]
    with open(os.path.join(AQUI, "elegidas.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print("bajadas:", len(filas), cuenta)


if __name__ == "__main__":
    main()
