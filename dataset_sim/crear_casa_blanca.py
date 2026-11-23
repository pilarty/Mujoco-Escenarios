"""
Arma la casa blanca: un departamento de paredes blancas con la luz de una
casa habitada, para probar al robot caminando.

Parte de procthor_real_01 (dormitorio, baño, cocina y living, paredes de
revoque liso pintado de blanco, 8 daños de pared) y cambia solo la luz:
  - en procthor_real_01 la luz de cada tramo de techo sale al azar (algunos
    apagados y otros al 35 %), asi que una misma sala queda mitad clara y
    mitad oscura, algo que no pasa en una casa real;
  - en la casa blanca todas las lamparas de techo estan prendidas y con la
    misma intensidad, como en una inspeccion con la luz de la casa encendida.
Las paredes, los muebles y los daños son los mismos.

Salida: escenarios/variantes/casa_blanca/

Uso (desde la carpeta Mujoco-Escenarios):
    python dataset_sim/crear_casa_blanca.py
"""
import json
import os
import re
import shutil

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGEN = "procthor_real_01"
NOMBRE = "casa_blanca"
LUZ_TECHO = (0.85, 0.85, 0.80)


def main():
    variantes = os.path.join(RAIZ, "escenarios", "variantes")
    destino = os.path.join(variantes, NOMBRE)
    if os.path.isdir(destino):
        shutil.rmtree(destino)
    shutil.copytree(os.path.join(variantes, ORIGEN), destino)

    ruta = os.path.join(destino, "escenario.xml")
    with open(ruta, encoding="utf-8") as f:
        xml = f.read()
    xml = xml.replace(f"/{ORIGEN}/", f"/{NOMBRE}/")
    r, g, b = LUZ_TECHO
    xml, cambiadas = re.subn(r'(<light name="luz_\d+"[^>]*?diffuse=")[^"]*(")', rf"\g<1>{r:.2f} {g:.2f} {b:.2f}\g<2>", xml)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(xml)

    ruta_v = os.path.join(destino, "variante.json")
    with open(ruta_v, encoding="utf-8") as f:
        variante = json.load(f)
    variante.update(nombre=NOMBRE, base=ORIGEN,
                    luz=f"{cambiadas} lamparas de techo prendidas con la misma intensidad {list(LUZ_TECHO)}")
    with open(ruta_v, "w", encoding="utf-8") as f:
        json.dump(variante, f, indent=2, ensure_ascii=False)
    print(f"{NOMBRE}: copia de {ORIGEN} con {cambiadas} lamparas de techo prendidas")


if __name__ == "__main__":
    main()
