"""
Run DRO grasp inference seeded with BODex's pre-generated grasp seeds.

For a given DGN scene_cfg .npy, this loads the matching BODex grasp .npy
(under sim_xhand/fc_left/DGN/graspdata/<obj>/<robot>/<stem>_grasp.npy),
converts the 20 stored seeds (trans + quat + 12 joint values) into DRO's
initial_q convention (trans + intrinsic-XYZ-euler + 18 joints in the order
hand.pk_chain expects), and feeds them as one batched run with batch_size=20.

Usage:
    python run_inference_DGN_seeded.py \
        --scene-cfg data/object/DGN_2k_origin/scene_cfg/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose004_0.npy \
        [--bodex-root /home/ubuntu/BODex/src/curobo/content/assets/output/sim_xhand/fc_left/DGN/graspdata] \
        [--hand xhand] [--debug]
"""

import argparse
import os
import sys

import numpy as np
import torch
import warnings
from hydra import compose, initialize_config_dir
from scipy.spatial.transform import Rotation

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference_DGN import GraspPoseProposal, load_dgn_scene_cfg


DEFAULT_BODEX_ROOT = (
    "/home/ubuntu/BODex/src/curobo/content/assets/output/sim_xhand/fc_left/DGN/graspdata"
)


def find_bodex_grasp_npy(scene_cfg_path: str, bodex_root: str) -> str:
    """Map data/object/DGN_2k_origin/scene_cfg/<obj>/<robot>/<stem>.npy
    -> <bodex_root>/<obj>/<robot>/<stem>_grasp.npy
    """
    scene_cfg_path = os.path.abspath(scene_cfg_path)
    parts = scene_cfg_path.split(os.sep)
    try:
        i = parts.index("scene_cfg")
    except ValueError as e:
        raise ValueError(
            f"scene_cfg path must contain 'scene_cfg/<obj>/<robot>/<stem>.npy', got {scene_cfg_path}"
        ) from e
    obj_name = parts[i + 1]
    robot_dir = parts[i + 2]
    stem = os.path.splitext(parts[i + 3])[0]

    candidates = [
        os.path.join(bodex_root, obj_name, robot_dir, f"{stem}_grasp.npy"),
        os.path.join(bodex_root, obj_name, robot_dir, f"{stem}__batch0grasp.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"No BODex grasp file found for {scene_cfg_path}. Tried: {candidates}"
    )


def convert_bodex_seed_to_dro(
    bodex_seed: np.ndarray,
    bodex_joint_names,
    dro_joint_names,
):
    """Convert BODex `seed` array (B, 19) = [tx,ty,tz, qw,qx,qy,qz, j0..j11]
    to DRO initial_q (B, 18) = [tx,ty,tz, ex,ey,ez, j0..j11] in DRO joint order.

    BODex stores left-hand joints (left_hand_*); DRO's xhand model uses
    right_hand_*. We map by the suffix after the first underscore so the same
    finger/joint lines up between the two.
    """
    bodex_seed = np.asarray(bodex_seed, dtype=np.float32)
    if bodex_seed.ndim != 2 or bodex_seed.shape[1] != 19:
        raise ValueError(
            f"expected bodex seed shape (B, 19), got {bodex_seed.shape}"
        )
    B = bodex_seed.shape[0]

    trans = bodex_seed[:, 0:3]                    # (B, 3)
    quat_wxyz = bodex_seed[:, 3:7]                # (B, 4) [w, x, y, z]
    joints = bodex_seed[:, 7:19]                  # (B, 12) in bodex order

    # scipy expects [x, y, z, w]
    quat_xyzw = np.stack(
        [quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]],
        axis=1,
    )
    euler = Rotation.from_quat(quat_xyzw).as_euler("XYZ").astype(np.float32)  # (B, 3)

    # Suffix-based mapping: drop "left_"/"right_" prefix to match by joint role.
    def strip_chirality(name: str) -> str:
        for pre in ("left_", "right_"):
            if name.startswith(pre):
                return name[len(pre):]
        return name

    bodex_idx_by_suffix = {strip_chirality(n): i for i, n in enumerate(bodex_joint_names)}

    # DRO joint order is [virtual_joint_x/y/z/roll/pitch/yaw, then 12 finger joints].
    # We only need the 12 finger joints in DRO's order; the first 6 come from trans+euler.
    finger_joint_names_in_dro = list(dro_joint_names[6:])
    if len(finger_joint_names_in_dro) != 12:
        raise ValueError(
            f"expected 12 finger joints, got {len(finger_joint_names_in_dro)}: {finger_joint_names_in_dro}"
        )

    permuted = np.zeros((B, 12), dtype=np.float32)
    for out_i, name in enumerate(finger_joint_names_in_dro):
        suffix = strip_chirality(name)
        if suffix not in bodex_idx_by_suffix:
            raise KeyError(
                f"DRO joint '{name}' (suffix '{suffix}') not present in BODex joint_names {bodex_joint_names}"
            )
        permuted[:, out_i] = joints[:, bodex_idx_by_suffix[suffix]]

    initial_q = np.concatenate([trans, euler, permuted], axis=1)  # (B, 18)
    return initial_q


def normalize_initial_q_translation(
    initial_q: np.ndarray,
    object_pc: np.ndarray,
):
    """DRO's pipeline shifts the object PC to zero-mean inside the network.
    We must shift the seed root translation by the same amount so the seeded
    hand pose stays aligned with the object PC the network sees.
    """
    object_pc_mean = object_pc.mean(axis=0)
    initial_q = initial_q.copy()
    initial_q[:, 0:3] -= object_pc_mean[None, :]
    return initial_q


def predict_q_to_world(
    predict_q,
    object_pc: np.ndarray,
    scene_cfg_pose,
):
    """Map predict_q from DRO's rotation-only, PC-zero-mean frame back into
    the BODex scene world frame.

    Inverse of the two translation shifts applied on the way in:
      world_t = q_t + object_pc_mean + scene_cfg_pose_t
    Wrist orientation and joint values are unchanged because we only shifted
    translations going in.
    """
    if torch.is_tensor(predict_q):
        out = predict_q.detach().cpu().numpy().copy()
    else:
        out = np.asarray(predict_q, dtype=np.float32).copy()
    object_pc_mean = object_pc.mean(axis=0)
    pose_t = np.asarray(scene_cfg_pose, dtype=np.float32).reshape(-1)[0:3]
    out[..., 0:3] = out[..., 0:3] + object_pc_mean[None, :] + pose_t[None, :]
    return out


def load_validate_cfg(batch_size: int, hand_name: str):
    """Compose the same config inference_DGN.py uses, then override batch_size."""
    config_dir = os.path.join(ROOT_DIR, "configs")
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        cfg = compose(
            config_name="validate",
            overrides=[
                f"dataset.batch_size={batch_size}",
                f"split_batch_size={batch_size}",
            ],
        )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene-cfg",
        default=os.path.join(
            ROOT_DIR,
            "data/object/DGN_2k_origin/scene_cfg/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose004_0.npy",
        ),
        help="Path to a DGN scene_cfg .npy",
    )
    ap.add_argument(
        "--bodex-root",
        default=DEFAULT_BODEX_ROOT,
        help="Root directory of BODex grasp .npy files (sim_xhand/fc_left/DGN/graspdata)",
    )
    ap.add_argument("--hand", default="xhand")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)

    bodex_npy = find_bodex_grasp_npy(args.scene_cfg, args.bodex_root)
    print(f"[seeded] scene_cfg = {args.scene_cfg}")
    print(f"[seeded] bodex_grasp = {bodex_npy}")

    bodex = np.load(bodex_npy, allow_pickle=True).item()
    seeds = np.asarray(bodex["seed"])              # (1, 20, 19)
    if seeds.ndim != 3 or seeds.shape[-1] != 19:
        raise ValueError(f"unexpected bodex seed shape: {seeds.shape}")
    seeds = seeds.reshape(-1, 19)                  # (20, 19)
    bodex_joint_names = list(bodex["joint_names"])
    batch_size = seeds.shape[0]

    cfg = load_validate_cfg(batch_size=batch_size, hand_name=args.hand)
    proposer = GraspPoseProposal(cfg)

    # Need DRO's joint ordering; trigger lazy creation of the hand model first.
    dro_joint_names = proposer.get_joint_order(args.hand)

    initial_q = convert_bodex_seed_to_dro(seeds, bodex_joint_names, dro_joint_names)

    info = load_dgn_scene_cfg(args.scene_cfg)
    if not os.path.exists(info["mesh_path"]):
        raise FileNotFoundError(info["mesh_path"])

    # prepare_data only applies the rotation part of object_pose so the PC
    # stays near zero-mean (matching the training distribution). The BODex
    # seed lives in the full world frame (R * mesh + t), so we subtract the
    # scene-cfg translation t from the seed root to put it in the same
    # rotation-only frame the network will see.
    pose_t = np.asarray(info["pose"], dtype=np.float32).reshape(-1)[0:3]
    initial_q[:, 0:3] -= pose_t[None, :]

    # Make sure the per-object point cloud is loaded so we can read its mean
    # and shift the seed translation by it (network internally zero-means).
    proposer.prepare_data(
        hand_name=args.hand,
        object_name=info["scene_id"],
        object_path=info["mesh_path"],
        object_scale=info["scale"],
        object_pose=info["pose"],
    )
    object_pc_np = proposer.object_pcs[info["scene_id"]].cpu().numpy()
    initial_q = normalize_initial_q_translation(initial_q, object_pc_np)

    print(f"[seeded] batch_size = {batch_size}, hand_dof = {len(dro_joint_names)}")
    out = proposer.predict_grasp_pose(
        hand_name=args.hand,
        object_name=info["scene_id"],
        object_path=info["mesh_path"],
        object_scale=info["scale"],
        object_pose=info["pose"],
        initial_q=torch.from_numpy(initial_q),
        debug=args.debug,
    )
    print("[seeded] predict_q shape:", tuple(out["predict_q"].shape))

    predict_q_world = predict_q_to_world(
        out["predict_q"], object_pc_np, info["pose"]
    )
    out["predict_q_world"] = predict_q_world
    print(
        f"[seeded] predict_q_world translation range: "
        f"min={predict_q_world[:, 0:3].min(0)}, max={predict_q_world[:, 0:3].max(0)}"
    )


if __name__ == "__main__":
    main()


"""
python run_inference_DGN_seeded.py \
    --scene-cfg data/object/DGN_2k_origin/scene_cfg/core_bottle_1a7ba1f4c892e2da30711cdbdbc73924/tabletop_ur10e/scale006_pose004_0.npy \
    --debug
"""