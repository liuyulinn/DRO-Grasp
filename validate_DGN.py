"""Validate DRO grasp predictions on DGN objects using IsaacGym.

For every grasp .npy under <out_dir>/<bodex_hand>/graspdata/<object_id>/
<scene_kind>/<scale_pose>_grasp.npy (the output of multi_gpu_inference_DGN.py),
this script:

  1. Pulls the GRASP stage (index 1 of pregrasp/grasp/squeeze) of the predicted
     hand pose. Shape (B, 7 + dof_b) = [tx, ty, tz, qw, qx, qy, qz, j0..j_b-1]
     in the world frame (object translated/rotated, table at z=-0.1).
  2. Reads the matching DGN scene_cfg .npy (referenced by `scene_path` in the
     grasp file) for the object's per-axis scale and world pose.
  3. Converts each world-frame hand pose into the object's NATIVE mesh frame by
     applying T_obj^{-1} to the hand root, and reorders the BODex finger joints
     into DRO's pk_chain order. Result: q_dro = [tx,ty,tz, ex,ey,ez, finger_dro].
  4. Runs validation/isaac_main_DGN.py as a subprocess. That helper bakes the
     per-axis scale into a temp URDF and runs the standard IsaacValidator with
     object at origin/identity, which is now consistent with our hand frame.

Usage:
  python validate_DGN.py
  python validate_DGN.py --out-dir dro_bodex_output --bodex-hand sim_xhand/fc_right
  python validate_DGN.py --object-glob 'core_bottle_1a7ba1f4*'  # smoke test

Mirrors validate_inference.py's logging style and writes the per-scene results
into validate_output/validate_DGN_<bodex_hand>.log.
"""

import argparse
import glob
import os
import subprocess
import sys
import warnings

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference_DGN import load_dgn_scene_cfg, DGN_PROCESSED_ROOT
from inference_DGN_bodex_seed import (
    DEFAULT_HAND_NAME_FOR_BODEX,
    build_joint_perm,
)
from utils.hand_model import create_hand_model


def _quat_wxyz_to_R(quat_wxyz):
    qw, qx, qy, qz = [float(x) for x in quat_wxyz]
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def world_grasp_to_dro_q(
    grasp_world,                  # (B, 7 + dof_b) [tx,ty,tz, qw,qx,qy,qz, j_bodex...]
    bodex_joint_names,            # list of length dof_b
    dro_joint_param_names_finger, # list of length dof_d (== dof_b expected)
    object_pose_world,            # (7,) [tx, ty, tz, qw, qx, qy, qz]
    src_side: str,
    dst_side: str,
):
    """Transform a batch of world-frame BODex grasp poses into DRO's q
    convention in the object's NATIVE mesh frame.

    DRO q_dro layout: [tx, ty, tz, ex, ey, ez, finger_q in DRO joint order].
    Native mesh frame = object at origin, identity rotation. That matches what
    isaac_main_DGN.py expects (it places the object at origin/identity).
    """
    grasp_world = np.asarray(grasp_world, dtype=np.float64)
    B = grasp_world.shape[0]

    obj_t = np.asarray(object_pose_world[:3], dtype=np.float64)
    obj_R = _quat_wxyz_to_R(object_pose_world[3:7])
    obj_R_inv = obj_R.T

    hand_t_world = grasp_world[:, 0:3]
    hand_R_world = R.from_quat(
        np.stack(
            [grasp_world[:, 4], grasp_world[:, 5], grasp_world[:, 6], grasp_world[:, 3]],
            axis=1,
        )
    ).as_matrix()  # (B, 3, 3)

    hand_t_native = (hand_t_world - obj_t[None, :]) @ obj_R_inv.T  # = R^T @ (t - t_obj)
    hand_R_native = obj_R_inv[None, :, :] @ hand_R_world           # (B, 3, 3)
    eulers_native = R.from_matrix(hand_R_native).as_euler("XYZ").astype(np.float32)

    perm = build_joint_perm(
        bodex_joint_names, dro_joint_param_names_finger, src_side, dst_side
    )
    finger_bodex = grasp_world[:, 7:]
    finger_dro = finger_bodex[:, perm].astype(np.float32)

    q_dro = np.zeros((B, 6 + len(dro_joint_param_names_finger)), dtype=np.float32)
    q_dro[:, 0:3] = hand_t_native.astype(np.float32)
    q_dro[:, 3:6] = eulers_native
    q_dro[:, 6:] = finger_dro
    return q_dro


def find_grasp_npys(graspdata_root: str, object_glob: str, scene_kind_glob: str):
    """Return [(object_id, scene_kind, scale_pose, npy_path), ...] sorted."""
    out = []
    for obj_dir in sorted(glob.glob(os.path.join(graspdata_root, object_glob))):
        if not os.path.isdir(obj_dir):
            continue
        object_id = os.path.basename(obj_dir)
        for scene_dir in sorted(glob.glob(os.path.join(obj_dir, scene_kind_glob))):
            if not os.path.isdir(scene_dir):
                continue
            scene_kind = os.path.basename(scene_dir)
            for npy in sorted(glob.glob(os.path.join(scene_dir, "*_grasp.npy"))):
                scale_pose = os.path.basename(npy).replace("_grasp.npy", "")
                out.append((object_id, scene_kind, scale_pose, npy))
    return out


def resolve_dgn_object_urdf(object_id: str, dgn_root: str = None) -> str:
    """Path to <dgn_root>/processed_data/<object_id>/urdf/coacd.urdf.

    Defaults to the in-repo DGN tree (data/object/DGN_2k_origin) when
    `dgn_root` is None, matching inference_DGN.DGN_PROCESSED_ROOT.
    """
    if dgn_root is None:
        return os.path.join(DGN_PROCESSED_ROOT, object_id, "urdf", "coacd.urdf")
    return os.path.join(dgn_root, "processed_data", object_id, "urdf", "coacd.urdf")


def remap_scene_cfg_path(orig_path: str, dgn_root: str) -> str:
    """Rewrite an embedded BODex `scene_path` against a user-provided DGN root.

    BODex stores absolute paths like
        .../DGN_2k_origin/scene_cfg/<obj_id>/<scene_kind>/<stem>.npy
    valid only on the machine that wrote them. We keep the trailing
    <obj_id>/<scene_kind>/<stem>.npy tail and join under <dgn_root>/scene_cfg/.
    """
    parts = orig_path.replace("\\", "/").split("/")
    if "scene_cfg" in parts:
        i = parts.index("scene_cfg")
        tail = parts[i + 1:]
    else:
        # Fallback: assume the last three components are <obj>/<scene_kind>/<stem>.npy
        tail = parts[-3:]
    return os.path.join(dgn_root, "scene_cfg", *tail)


def call_isaac_main_dgn(
    robot_name: str,
    object_urdf: str,
    object_tag: str,
    scale_xyz,
    q_batch: torch.Tensor,
    gpu: int,
):
    """Subprocess wrapper around validation/isaac_main_DGN.py to dodge
    IsaacGym's GPU-memory leak across scenes."""
    tmp_dir = os.path.join(ROOT_DIR, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    q_file = os.path.join(tmp_dir, f"q_dgn_validate_{gpu}.pt")
    ret_file = os.path.join(tmp_dir, f"isaac_main_DGN_ret_{gpu}.pt")
    if os.path.exists(ret_file):
        os.remove(ret_file)
    torch.save(q_batch, q_file)

    cmd = [
        "python",
        os.path.join(ROOT_DIR, "validation/isaac_main_DGN.py"),
        "--robot_name", robot_name,
        "--object_urdf", object_urdf,
        "--object_tag", object_tag,
        "--scale_x", f"{float(scale_xyz[0])}",
        "--scale_y", f"{float(scale_xyz[1])}",
        "--scale_z", f"{float(scale_xyz[2])}",
        "--q_file", q_file,
        "--ret_file", ret_file,
        "--gpu", str(gpu),
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    try:
        save_data = torch.load(ret_file)
        success = save_data["success"]
        q_isaac = save_data["q_isaac"]
    except FileNotFoundError as e:
        cprint(f"[isaac_main_DGN] missing return file: {e}", "yellow")
        cprint(ret.stdout.strip(), "blue")
        cprint(ret.stderr.strip(), "red")
        return None, None
    finally:
        if os.path.exists(q_file):
            os.remove(q_file)
        if os.path.exists(ret_file):
            os.remove(ret_file)
    return success, q_isaac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=os.path.join(ROOT_DIR, "dro_bodex_output"),
                   help="root directory written by multi_gpu_inference_DGN.py")
    p.add_argument("--bodex-hand", default="sim_xhand/fc_right")
    p.add_argument("--scene-kind", default="*",
                   help="glob over scene_kind subfolders (default: all, e.g. tabletop_ur10e)")
    p.add_argument("--object-glob", default="*",
                   help="glob over object_id directories")
    p.add_argument("--hand-name", default=None,
                   help="DRO hand_name; auto-inferred from --bodex-hand sim_* prefix")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--obj-start", type=int, default=None)
    p.add_argument("--obj-end", type=int, default=None)
    p.add_argument("--log-name", default=None,
                   help="basename for the per-scene log (default: validate_DGN_<bodex_hand>)")
    p.add_argument(
        "--dgn-root", default=None,
        help="Override the DGN_2k_origin root. Used to (a) remap the absolute "
             "scene_path embedded in each grasp .npy onto <dgn_root>/scene_cfg/... "
             "and (b) load object URDFs from <dgn_root>/processed_data/<obj>/urdf/. "
             "Defaults to inference_DGN.DGN_PROCESSED_ROOT (the in-repo data tree).",
    )
    args = p.parse_args()

    sim_dir = args.bodex_hand.split("/")[0]
    side_dir = args.bodex_hand.split("/")[1] if "/" in args.bodex_hand else "fc_right"
    src_side = "left" if side_dir.endswith("left") else "right"
    dst_side = "right"  # all DRO URDFs in this repo are right-handed
    hand_name = args.hand_name or DEFAULT_HAND_NAME_FOR_BODEX.get(sim_dir)
    if hand_name is None:
        raise SystemExit(
            f"could not infer DRO hand_name from --bodex-hand={args.bodex_hand}; "
            f"pass --hand-name=..."
        )

    graspdata_root = os.path.join(args.out_dir, args.bodex_hand, "graspdata")
    if not os.path.isdir(graspdata_root):
        raise SystemExit(f"graspdata root not found: {graspdata_root}")

    entries = find_grasp_npys(graspdata_root, args.object_glob, args.scene_kind)
    if not entries:
        raise SystemExit(
            f"no *_grasp.npy under {graspdata_root} "
            f"(object_glob={args.object_glob!r}, scene_kind={args.scene_kind!r})"
        )

    if args.obj_start is not None or args.obj_end is not None:
        s = 0 if args.obj_start is None else args.obj_start
        e = len(entries) if args.obj_end is None else args.obj_end
        entries = entries[s:e]

    log_dir = os.path.join(ROOT_DIR, "validate_output")
    os.makedirs(log_dir, exist_ok=True)
    log_name = args.log_name or f"validate_DGN_{args.bodex_hand.replace('/', '_')}"
    log_path = os.path.join(log_dir, f"{log_name}.log")
    print(f"[validate_DGN] hand={hand_name} bodex_hand={args.bodex_hand} "
          f"scenes={len(entries)} log={log_path}")

    dro_hand = create_hand_model(hand_name, device=torch.device("cpu"))
    dro_joint_param_names_finger = dro_hand.pk_chain.get_joint_parameter_names()[6:]

    per_object_succ = {}   # object_id -> [succ, total]
    total_succ = 0
    total_num = 0

    for object_id, scene_kind, scale_pose, npy in entries:
        raw = np.load(npy, allow_pickle=True).item()
        robot_pose = np.asarray(raw["robot_pose"])  # (1, B, 3, 7+dof_b)
        if robot_pose.ndim != 4 or robot_pose.shape[2] != 3:
            cprint(f"[skip] bad robot_pose shape {robot_pose.shape} in {npy}", "yellow")
            continue
        grasp_world = robot_pose[0, :, 1, :].astype(np.float32)  # stage 1 = grasp
        bodex_joint_names = list(raw["joint_names"])

        scene_paths = list(raw.get("scene_path", []))
        if not scene_paths:
            cprint(f"[skip] no scene_path in {npy}", "yellow")
            continue
        scene_cfg_path = str(scene_paths[0])
        if args.dgn_root is not None:
            scene_cfg_path = remap_scene_cfg_path(scene_cfg_path, args.dgn_root)
        if not os.path.exists(scene_cfg_path):
            cprint(f"[skip] scene_cfg not on disk: {scene_cfg_path}", "yellow")
            continue
        info = load_dgn_scene_cfg(scene_cfg_path)
        object_urdf = resolve_dgn_object_urdf(object_id, args.dgn_root)
        if not os.path.exists(object_urdf):
            cprint(f"[skip] urdf missing: {object_urdf}", "yellow")
            continue

        try:
            q_dro = world_grasp_to_dro_q(
                grasp_world,
                bodex_joint_names,
                dro_joint_param_names_finger,
                info["pose"],
                src_side,
                dst_side,
            )
        except ValueError as e:
            cprint(f"[skip] joint mapping failed for {object_id}/{scale_pose}: {e}",
                   "yellow")
            continue

        q_batch = torch.from_numpy(q_dro)
        scale_xyz = np.asarray(info["scale"], dtype=np.float32).reshape(-1)
        if scale_xyz.size == 1:
            scale_xyz = np.array([scale_xyz[0]] * 3, dtype=np.float32)
        elif scale_xyz.size != 3:
            cprint(
                f"[skip] unexpected scale shape {scale_xyz.shape} for "
                f"{object_id}/{scale_pose}",
                "yellow",
            )
            continue

        object_tag = f"{object_id}/{scene_kind}/{scale_pose}"
        success, _ = call_isaac_main_dgn(
            robot_name=hand_name,
            object_urdf=object_urdf,
            object_tag=object_tag,
            scale_xyz=scale_xyz,
            q_batch=q_batch,
            gpu=args.gpu,
        )
        n = q_batch.shape[0]
        n_succ = int(success.sum().item()) if success is not None else -1
        cprint(
            f"[{hand_name}/{object_tag}] {n_succ}/{n} "
            f"({(n_succ / n * 100) if n_succ >= 0 else float('nan'):.2f}%)",
            "green",
        )
        with open(log_path, "a") as f:
            print(f"[{hand_name}/{object_tag}] {n_succ}/{n}", file=f)

        if n_succ >= 0:
            per_object_succ.setdefault(object_id, [0, 0])
            per_object_succ[object_id][0] += n_succ
            per_object_succ[object_id][1] += n
            total_succ += n_succ
            total_num += n

    if total_num == 0:
        cprint("[validate_DGN] no successful runs", "red")
        return

    for object_id, (succ, n) in per_object_succ.items():
        cprint(
            f"[{hand_name}/{object_id}] {succ}/{n} ({succ / n * 100:.2f}%)",
            "magenta",
        )
        with open(log_path, "a") as f:
            print(f"[{hand_name}/{object_id}] {succ}/{n}", file=f)

    cprint(
        f"[TOTAL {hand_name}] {total_succ}/{total_num} "
        f"({total_succ / total_num * 100:.2f}%)",
        "yellow",
    )
    with open(log_path, "a") as f:
        print(f"[TOTAL {hand_name}] {total_succ}/{total_num}", file=f)


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)
    main()
