"""Viser viewer for BODex-format grasp .npy files written by inference_DGN_bodex_seed.py.

Each .npy holds robot_pose (1, B, S, 7 + dof_bodex) in the *world* frame
(xyz + wxyz quat + BODex-ordered finger joints), plus the world_cfg describing the
object mesh (file, scale, pose) and the table cuboid, so the whole scene can be
rebuilt without touching the original scene_cfg.

Usage:
  uv run python visualization/vis_bodex_output.py \
      --root dro_bodex_output/sim_backallegro/fc_left/graspdata \
      --hand-name allegro_left
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model

# BODex hand tag -> DRO hand_name, same table as inference_DGN_bodex_seed.py
HAND_NAME_FROM_BODEX = {
    "sim_xhand": "xhand_left",
    "sim_fixsharpa": "fixsharpa_left",
    "sim_backallegro": "allegro_left",
}

STAGE_NAMES = ["pregrasp", "grasp", "squeeze"]


def parse_vec(value):
    """world_cfg numeric fields are sometimes stored as str(np.ndarray)."""
    if isinstance(value, str):
        return np.fromstring(value.strip("[]").replace("\n", " "), sep=" ")
    return np.asarray(value, dtype=np.float64)


def pose_to_matrix(pose):
    """pose = [x, y, z, qw, qx, qy, qz] -> 4x4 homogeneous matrix."""
    pose = parse_vec(pose)
    m = np.eye(4)
    m[:3, :3] = R.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    m[:3, 3] = pose[:3]
    return m


def bodex_row_to_dro_q(row, perm):
    """(7 + dof_b,) world-frame BODex pose -> (6 + dof_d,) DRO q (xyz + intrinsic XYZ euler)."""
    row = np.asarray(row, dtype=np.float64)
    euler = R.from_quat(row[[4, 5, 6, 3]]).as_euler("XYZ")
    finger = row[7:][perm]
    return torch.tensor(np.concatenate([row[:3], euler, finger]), dtype=torch.float32)


def build_perm(bodex_joint_names, dro_joint_names_finger):
    """perm[i] = index into the BODex finger vector for DRO finger joint i."""
    return [bodex_joint_names.index(jn) for jn in dro_joint_names_finger]


def load_npy(path):
    raw = np.load(path, allow_pickle=True)
    return raw.item() if raw.dtype == object and raw.shape == () else raw


def resolve_mesh_path(file_path):
    """world_cfg mesh paths contain '../..' segments; also try the sibling normalized.obj."""
    p = os.path.normpath(file_path)
    if os.path.exists(p):
        return p
    alt = os.path.join(os.path.dirname(p), "normalized.obj")
    return alt if os.path.exists(alt) else None


def object_meshes_from_world_cfg(world_cfg):
    """Return [(name, trimesh)] for every mesh entry, already scaled and posed."""
    out = []
    cfg = world_cfg[0] if isinstance(world_cfg, (list, tuple)) else world_cfg
    for name, spec in cfg.get("mesh", {}).items():
        path = resolve_mesh_path(spec["file_path"])
        if path is None:
            print(f"[warn] object mesh not found: {spec['file_path']}")
            continue
        mesh = trimesh.load_mesh(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.apply_scale(parse_vec(spec["scale"]))
        mesh.apply_transform(pose_to_matrix(spec["pose"]))
        out.append((name.split("/")[-1], mesh))
    return out


def table_mesh_from_world_cfg(world_cfg):
    cfg = world_cfg[0] if isinstance(world_cfg, (list, tuple)) else world_cfg
    for name, spec in cfg.get("cuboid", {}).items():
        box = trimesh.creation.box(extents=parse_vec(spec["dims"]))
        box.apply_transform(pose_to_matrix(spec["pose"]))
        return name, box
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="dro_bodex_output/sim_backallegro/fc_left/graspdata",
        help="graspdata/ folder holding <object_id>/<scene_kind>/*_grasp.npy",
    )
    p.add_argument("--hand-name", default=None, help="DRO hand_name (inferred from --root if omitted)")
    p.add_argument("--scene-kind", default="tabletop_ur10e")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    root = args.root if os.path.isabs(args.root) else os.path.join(ROOT_DIR, args.root)
    if not os.path.isdir(root):
        raise SystemExit(f"root not found: {root}")

    hand_name = args.hand_name
    if hand_name is None:
        for tag, name in HAND_NAME_FROM_BODEX.items():
            if tag in root:
                hand_name = name
                break
    if hand_name is None:
        raise SystemExit(f"could not infer hand_name from {root}; pass --hand-name")

    def list_objects():
        return sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d, args.scene_kind))
        )

    object_ids = list_objects()
    if not object_ids:
        raise SystemExit(f"no objects with a {args.scene_kind}/ subdir under {root}")

    print(f"[info] {len(object_ids)} objects under {root}")
    print(f"[info] loading hand model '{hand_name}' (cpu)")
    hand = create_hand_model(hand_name, device=torch.device("cpu"))
    dro_finger_names = hand.pk_chain.get_joint_parameter_names()[6:]

    def scenes_for(object_id):
        return sorted(
            os.path.basename(f).replace("_grasp.npy", "")
            for f in glob.glob(os.path.join(root, object_id, args.scene_kind, "*_grasp.npy"))
        )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True

    object_dd = server.gui.add_dropdown("object", options=object_ids, initial_value=object_ids[0])
    scene_dd = server.gui.add_dropdown("scene", options=scenes_for(object_ids[0]))
    grasp_slider = server.gui.add_slider("grasp_idx", min=0, max=0, step=1, initial_value=0)
    stage_dd = server.gui.add_dropdown("stage", options=STAGE_NAMES, initial_value="grasp")
    show_seed = server.gui.add_checkbox("show BODex seed", initial_value=False)
    show_table = server.gui.add_checkbox("show table", initial_value=True)
    # Generation may still be writing into `root`, so allow re-enumerating on demand.
    rescan_btn = server.gui.add_button("rescan objects")
    info_md = server.gui.add_markdown("")

    # cache: npy path -> (data dict, object meshes, table mesh, perm)
    cache = {}

    def load_scene(object_id, scene_name):
        path = os.path.join(root, object_id, args.scene_kind, f"{scene_name}_grasp.npy")
        if path not in cache:
            data = load_npy(path)
            perm = build_perm(list(data["joint_names"]), dro_finger_names)
            cache[path] = (
                data,
                object_meshes_from_world_cfg(data["world_cfg"]),
                table_mesh_from_world_cfg(data["world_cfg"]),
                perm,
            )
        return path, cache[path]

    def add_hand(node_name, row, perm, color, visible=True):
        q = bodex_row_to_dro_q(row, perm)
        mesh = hand.get_trimesh_q(q)["visual"]
        server.scene.add_mesh_simple(
            node_name, mesh.vertices, mesh.faces,
            color=color, opacity=0.85, visible=visible,
        )

    def redraw():
        object_id, scene_name = object_dd.value, scene_dd.value
        path, (data, obj_meshes, (table_name, table), perm) = load_scene(object_id, scene_name)

        robot_pose = data["robot_pose"][0]              # (B, S, 7 + dof_b)
        n_grasps, n_stages = robot_pose.shape[0], robot_pose.shape[1]
        if grasp_slider.max != n_grasps - 1:
            grasp_slider.max = n_grasps - 1
        idx = min(int(grasp_slider.value), n_grasps - 1)
        stage = min(STAGE_NAMES.index(stage_dd.value), n_stages - 1)

        server.scene.reset()
        server.scene.world_axes.visible = True

        for name, mesh in obj_meshes:
            server.scene.add_mesh_simple(
                f"/object/{name}", mesh.vertices, mesh.faces,
                color=(239, 132, 167), opacity=1.0,
            )
        if table is not None:
            server.scene.add_mesh_simple(
                f"/{table_name}", table.vertices, table.faces,
                color=(200, 200, 200), opacity=0.6, visible=show_table.value,
            )

        add_hand("/hand_predict", robot_pose[idx, stage], perm, (102, 192, 255))
        if show_seed.value and "seed" in data:
            add_hand("/hand_seed", data["seed"][0][idx], perm, (160, 160, 160))

        info_md.content = (
            f"**{object_id}**\n\n"
            f"scene `{scene_name}` &middot; grasp {idx}/{n_grasps - 1} &middot; "
            f"stage `{STAGE_NAMES[stage]}`\n\n"
            f"`{os.path.relpath(path, ROOT_DIR)}`"
        )

    def on_object_change(_):
        names = scenes_for(object_dd.value)
        scene_dd.options = names
        if names:
            scene_dd.value = names[0]
        redraw()

    def on_rescan(_):
        names = list_objects()
        object_dd.options = names
        current = object_dd.value
        scene_dd.options = scenes_for(current) if current in names else []
        print(f"[info] rescan: {len(names)} objects")
        redraw()

    object_dd.on_update(on_object_change)
    rescan_btn.on_click(on_rescan)
    for handle in (scene_dd, grasp_slider, stage_dd, show_seed, show_table):
        handle.on_update(lambda _: redraw())

    redraw()
    print(f"[info] viser server on http://{args.host}:{args.port}")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
