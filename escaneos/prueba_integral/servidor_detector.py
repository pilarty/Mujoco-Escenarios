"""Servidor local con el detector de danos del back (mismo codigo, pesos, temperatura y umbral) para la prueba integral.

La navegacion corre en el .venv de este proyecto, que no tiene TensorFlow; este servidor corre con el Python del back y contesta, para cada
foto o recorte en PNG, si hay dano y con que probabilidad.

Uso: python escaneos/prueba_integral/servidor_detector.py ../g1-inspection-backend 8010
"""
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

sys.path.insert(0, os.path.abspath(sys.argv[1]))
from domain import detector_danos_cnn

detector_danos_cnn.cargar()


class Manejador(BaseHTTPRequestHandler):
    def do_POST(self):
        datos = self.rfile.read(int(self.headers["Content-Length"]))
        imagen = Image.open(io.BytesIO(datos)).convert("RGB").resize((224, 224))
        hay, prob = detector_danos_cnn.hay_dano(imagen)
        cuerpo = json.dumps({"hay": bool(hay), "prob": float(prob)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def log_message(self, *args):
        pass


print(f"detector listo: {detector_danos_cnn.MODELO_DETECCION_PATH.name}, temperatura {detector_danos_cnn.TEMPERATURA}, "
      f"umbral {detector_danos_cnn.UMBRAL_DANIO}", flush=True)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), Manejador).serve_forever()
