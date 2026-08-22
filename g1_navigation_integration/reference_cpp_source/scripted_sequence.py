import time

CMD_FILE = "/tmp/scripted_joystick_cmd.txt"

def write_cmd(lt=0, rt=0, up=0, b=0, a=0):
    with open(CMD_FILE, "w") as f:
        f.write(f"{lt} {rt} {up} {b} {a}\n")

def clear():
    write_cmd()

clear()  # aseguramos estado limpio antes de arrancar, sin importar corridas previas

print("Esperando 1s de margen antes de arrancar...")
time.sleep(1)

print("Passive -> FixStand: sosteniendo LT + tocando up")
write_cmd(lt=1)
time.sleep(0.5)          # deja que LT.pressed cruce el umbral del suavizado
write_cmd(lt=1, up=1)
time.sleep(0.2)
clear()

print("Esperando 3s para que FixStand termine su interpolacion de 2s...")
time.sleep(3.0)

print("FixStand -> Velocity: sosteniendo RT + tocando A")
write_cmd(rt=1)
time.sleep(0.5)
write_cmd(rt=1, a=1)
time.sleep(0.2)
clear()

print("Secuencia completa.")
