"""
Servidor local para exponer la carpeta 'reporte/' (detections.json + fotos)
a la interfaz web, que corre en otro repo/otro puerto.

Uso:
    python serve_reporte.py

Por defecto sirve en http://localhost:8000
Ahí van a quedar disponibles:
    http://localhost:8000/detections.json
    http://localhost:8000/images/frame_00300.png
"""

import os
import http.server
import socketserver

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTE_DIR = os.path.join(PROJECT_ROOT, "simulaciones", "g1_saca_fotos", "reporte")

PUERTO = 8000


class HandlerConCORS(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=REPORTE_DIR, **kwargs)

    def end_headers(self):
        # Habilita que la interfaz web (en otro origen/puerto) pueda
        # leer detections.json y las imágenes sin que el navegador lo bloquee.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    os.makedirs(REPORTE_DIR, exist_ok=True)
    with socketserver.TCPServer(("", PUERTO), HandlerConCORS) as httpd:
        print(f"Sirviendo {REPORTE_DIR}")
        print(f"-> http://localhost:{PUERTO}/detections.json")
        httpd.serve_forever()