"""Merge DGN multi-link coacd.urdf into a single-link urdf for DRO Isaac validator."""
import os
import sys
import xml.etree.ElementTree as ET

DGN = "/home/yulin/workspace/DexCom/assets/misc/DGN_2k_origin/processed_data"
DST = "/home/yulin/workspace/DRO-Grasp/data/data_urdf/object/dgn"


def merge(obj_id: str, scale: float = 1.0):
    src = f"{DGN}/{obj_id}/urdf/coacd.urdf"
    out_dir = f"{DST}/{obj_id}"
    os.makedirs(out_dir, exist_ok=True)

    tree = ET.parse(src)
    root = tree.getroot()

    if scale != 1.0:
        s = f"{scale} {scale} {scale}"
        for m in root.iter("mesh"):
            m.set("scale", s)

    visuals, collisions = [], []
    for link in list(root.findall("link")):
        visuals.extend(link.findall("visual"))
        collisions.extend(link.findall("collision"))
        root.remove(link)
    for joint in list(root.findall("joint")):
        root.remove(joint)

    new_link = ET.SubElement(root, "link", attrib={"name": "object"})
    inertial = ET.SubElement(new_link, "inertial")
    ET.SubElement(inertial, "origin", attrib={"xyz": "0 0 0", "rpy": "0 0 0"})
    mass = ET.SubElement(inertial, "mass", attrib={"value": "0.1"})
    ET.SubElement(inertial, "inertia", attrib={
        "ixx": "1e-3", "ixy": "0", "ixz": "0",
        "iyy": "1e-3", "iyz": "0", "izz": "1e-3",
    })
    for v in visuals:
        new_link.append(v)
    for c in collisions:
        new_link.append(c)

    out_urdf = f"{out_dir}/coacd_decomposed_object_one_link.urdf"
    tree.write(out_urdf, xml_declaration=True, encoding="utf-8")

    meshes_link = f"{out_dir}/meshes"
    if os.path.lexists(meshes_link):
        os.remove(meshes_link)
    os.symlink(f"{DGN}/{obj_id}/urdf/meshes", meshes_link)
    print(f"[ok] {out_urdf}")


if __name__ == "__main__":
    args = sys.argv[1:]
    scale = 1.0
    if args and args[0].startswith("--scale="):
        scale = float(args[0].split("=", 1)[1])
        args = args[1:]
    ids = args or ["core_bottle_1a7ba1f4c892e2da30711cdbdbc73924"]
    for i in ids:
        merge(i, scale=scale)
