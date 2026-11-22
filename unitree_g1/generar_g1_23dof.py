"""Genera unitree_g1/g1_23dof.xml a partir de unitree_g1/g1.xml (29 DOF).

Pasa del G1 de 29 DOF al G1 EDU Standard de 23 DOF, copiando la estructura y las
inertias del modelo con el que se entrena (unitree_rl_mjlab, g1_23dof.xml):
  - cintura: solo waist_yaw (sin waist_roll ni waist_pitch); torso_link cuelga directo del pelvis
  - munecas: solo wrist_roll; el cuerpo de la muneca es 'wrist_roll_rubber_hand' (mano de goma incluida)
Uso: python mk23.py g1.xml g1_23dof.xml
"""
import sys
import xml.etree.ElementTree as ET

src, dst = sys.argv[1], sys.argv[2]
tree = ET.parse(src)
root = tree.getroot()
root.set("model", "g1_23dof_rev_1_0")

REMOVE_JOINTS = ["waist_roll_joint", "waist_pitch_joint",
                 "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                 "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
order = [j.get("name") for j in root.iter("joint") if j.get("name")]   # orden de qpos antes de borrar


def parent_map():
    return {c: p for p in root.iter() for c in p}


def find_body(name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    raise KeyError(name)


# ---- cintura: pelvis > waist_yaw_link > waist_roll_link > torso_link  ==>  pelvis > torso_link (waist_yaw)
yaw_link = find_body("waist_yaw_link")
roll_link = find_body("waist_roll_link")
torso = find_body("torso_link")
pm = parent_map()
pelvis = pm[yaw_link]
yaw_joint = [j for j in yaw_link.findall("joint") if j.get("name") == "waist_yaw_joint"][0]
for j in torso.findall("joint"):
    torso.remove(j)                                  # waist_pitch_joint
yaw_joint.set("pos", "0 0 0")
torso.insert(1, yaw_joint)                           # despues del <inertial>
torso.set("pos", roll_link.get("pos"))               # -0.0039635 0 0.044
idx = list(pelvis).index(yaw_link)
pelvis.remove(yaw_link)
pelvis.insert(idx, torso)
# la malla del torso de 23 DOF esta en el marco del pelvis (igual que logo y cabeza)
for g in torso.findall("geom"):
    if g.get("mesh") == "torso_link":
        g.set("pos", "0.0039635 0 -0.044")

# ---- munecas: wrist_roll_link (con pitch y yaw colgando) ==> wrist_roll_rubber_hand
HAND = {
    "left": dict(pos="0.1 0.00188791 -0.01",
                 inertial=dict(pos="0.107947 0.00163512 0.00202245", quat="0.494051 0.504265 0.48416 0.516933"),
                 capsule="0.02 0 0 0.22 -0.02 0"),
    "right": dict(pos="0.1 -0.00188791 -0.01",
                  inertial=dict(pos="0.107947 -0.00163512 0.00202245", quat="0.516933 0.48416 0.504265 0.494051"),
                  capsule="0.02 0 0 0.22 0.02 0"),
}
for side, h in HAND.items():
    wr = find_body(f"{side}_wrist_roll_link")
    elbow = parent_map()[wr]
    joint = wr.find("joint")
    new = ET.Element("body", name=f"{side}_wrist_roll_rubber_hand", pos=h["pos"])
    ET.SubElement(new, "inertial", pos=h["inertial"]["pos"], quat=h["inertial"]["quat"],
                  mass="0.356929", diaginertia="0.00200292 0.0019426 0.000195232")
    new.append(joint)
    ET.SubElement(new, "geom", {"class": "visual", "mesh": f"{side}_wrist_roll_rubber_hand"})
    ET.SubElement(new, "site", name=f"{side}_palm", pos="0.08 0 0", size="0.01")
    ET.SubElement(new, "geom", {"class": "collision", "name": f"{side}_hand_collision", "type": "capsule",
                                "size": "0.035", "fromto": h["capsule"], "rgba": ".2 .6 .2 .2"})
    i = list(elbow).index(wr)
    elbow.remove(wr)
    elbow.insert(i, new)

# ---- mallas
asset = root.find("asset")
for m in list(asset.findall("mesh")):
    f = m.get("file")
    n = m.get("name") or f.rsplit(".", 1)[0]
    if n in ("waist_yaw_link", "waist_roll_link") or "wrist_pitch" in n or "wrist_yaw" in n \
            or n in ("left_wrist_roll_link", "right_wrist_roll_link", "left_rubber_hand", "right_rubber_hand"):
        asset.remove(m)
    elif n == "torso_link":
        m.set("file", "torso_link_23dof_rev_1_0.STL")
for side in ("left", "right"):
    ET.SubElement(asset, "mesh", name=f"{side}_wrist_roll_rubber_hand", file=f"{side}_wrist_roll_rubber_hand.STL")

# ---- actuadores
act = root.find("actuator")
for c in list(act):
    if c.get("name") in REMOVE_JOINTS:
        act.remove(c)

# ---- contactos: en entrenamiento las piernas se modelan con capsulas finas y la mano no las toca; con las mallas
# de colision de las piernas de este XML la mano queda rozando la cadera en la pose por defecto y empuja al robot.
# Se excluye la mano contra el pelvis y los eslabones de la cadera del mismo lado (con el entorno sigue chocando).
contact = ET.SubElement(root, "contact")
for side in ("left", "right"):
    for other in ("pelvis", f"{side}_hip_pitch_link", f"{side}_hip_roll_link", f"{side}_hip_yaw_link"):
        ET.SubElement(contact, "exclude", body1=f"{side}_wrist_roll_rubber_hand", body2=other)

# ---- keyframe
keep = [i for i, n in enumerate(order) if n not in REMOVE_JOINTS]
for k in root.iter("key"):
    q, c = k.get("qpos").split(), k.get("ctrl").split()
    k.set("qpos", " ".join(q[:7] + [q[7 + i] for i in keep]))
    k.set("ctrl", " ".join(c[i] for i in keep))

ET.indent(tree)
tree.write(dst, encoding="utf-8")
print("articulaciones:", len(keep))
