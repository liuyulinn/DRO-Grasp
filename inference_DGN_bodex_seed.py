"""Run DRO inference on DGN scenes seeded with BODex grasp initializations,
saving predictions in BODex's dict-of-arrays .npy format.

For every BODex grasp .npy under
    /home/ubuntu/BODex/.../output/<sim_hand>/<fc_side>/<obj_set>/graspdata/
        <object_id>/<scene_kind>/<scale_pose_id>_grasp.npy
this script:

  1. Maps the recorded scene back to a DGN scene_cfg .npy (BODex's scene_path
     field already points to it) so DRO sees the same object mesh, scale, and
     pose-on-table.
  2. Converts each BODex `seed` row (xyz + wxyz_quat + joint_q in BODex joint
     order) into DRO's initial_q convention (xyz + intrinsic-XYZ euler +
     joint_q in DRO's pk_chain joint order). Joint names are mapped left<->right
     to match DRO's right-hand URDFs.
  3. Calls GraspPoseProposal.predict_grasp_pose with that seed as initial_q.
  4. Saves predictions back in BODex's dict format -- one .npy per scene with
     `robot_pose`, `seed`, `joint_names`, `world_cfg`, etc. -- into a tree
     that mirrors BODex's `graspdata/.../<scale_pose>_grasp.npy` layout:

        {out_dir}/<bodex_hand>/<obj_set>/graspdata/
            <object_id>/<scene_kind>/<scale_pose>_grasp.npy

     Joint values are written back in BODex joint order (left/right side
     restored) so the file is a drop-in replacement for the BODex grasp.

Usage:
  python inference_DGN_bodex_seed.py +bodex_hand=sim_xhand/fc_left
  python inference_DGN_bodex_seed.py +bodex_hand=sim_xhand/fc_left \
      +bodex_object_set=DGN +bodex_scene_kind=tabletop_ur10e \
      +bodex_glob='core_bottle_*'
  python inference_DGN_bodex_seed.py +bodex_hand=sim_fixsharpa/fc_right \
      hand_name=sharpa
"""

import os
import sys
import glob
import json
import warnings

import hydra
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference_DGN import GraspPoseProposal, load_dgn_scene_cfg, DGN_SCENE_CFG_ROOT


BODEX_OUTPUT_ROOT = "/home/ubuntu/BODex/src/curobo/content/assets/output"

# BODex hand tag -> DRO hand_name in urdf_assets_meta.json
DEFAULT_HAND_NAME_FOR_BODEX = {
    "sim_xhand": "xhand",
    "sim_fixsharpa": "sharpa",
    "sim_backallegro": "allegro",
}


def _swap_side(name: str, src_side: str, dst_side: str) -> str:
    """left_hand_thumb -> right_hand_thumb, etc. Substring-anchored."""
    src = src_side + "_"
    dst = dst_side + "_"
    if name.startswith(src):
        return dst + name[len(src):]
    return name


def build_joint_perm(bodex_joint_names, dro_joint_param_names, src_side, dst_side):
    """Permutation from BODex joint order -> DRO joint_parameter_names[6:].

    Returns a list `perm` such that q_dro_finger[i] = q_bodex[perm[i]]. Raises
    if any DRO finger joint cannot be found in BODex names after side-swap.
    """
    bodex_swapped = [_swap_side(n, src_side, dst_side) for n in bodex_joint_names]
    perm = []
    missing = []
    for jn in dro_joint_param_names:
        try:
            perm.append(bodex_swapped.index(jn))
        except ValueError:
            missing.append(jn)
    if missing:
        raise ValueError(
            f"DRO joint(s) not found in BODex names after {src_side}->{dst_side} "
            f"swap: {missing}\nBODex (swapped): {bodex_swapped}\nDRO: {dro_joint_param_names}"
        )
    return perm


def bodex_seed_to_dro_initial_q(
    bodex_seed_row,
    bodex_joint_names,
    dro_joint_param_names_finger,  # DRO joints excluding the 6 virtual ones
    object_translation_world,      # (3,) translation applied to the object in scene_cfg
    src_side,
    dst_side,
):
    """Convert a single BODex seed row (7 + n_b joints) -> DRO initial_q (6 + n_d joints).

    BODex base pose lives in the world frame where the object also has translation
    `object_translation_world`. DRO's network expects an object-centered point
    cloud (translation removed but rotation kept, see inference_DGN.py:117-132),
    so we subtract that translation from the hand base before passing it as the
    initial guess.
    """
    bodex_seed_row = np.asarray(bodex_seed_row, dtype=np.float64)
    base_xyz = bodex_seed_row[:3] - np.asarray(object_translation_world, dtype=np.float64)
    qw, qx, qy, qz = bodex_seed_row[3:7]
    rot_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
    euler_xyz = R.from_matrix(rot_mat).as_euler("XYZ")  # intrinsic, matches utils/rotation.py

    bodex_finger_q = bodex_seed_row[7:]
    perm = build_joint_perm(
        bodex_joint_names, dro_joint_param_names_finger, src_side, dst_side
    )
    dro_finger_q = bodex_finger_q[perm]

    initial_q = np.concatenate([base_xyz, euler_xyz, dro_finger_q]).astype(np.float32)
    return initial_q


def load_bodex_npy(npy_path):
    raw = np.load(npy_path, allow_pickle=True)
    if raw.dtype == object and raw.shape == ():
        raw = raw.item()
    return raw


def resolve_scene_cfg_path(bodex_data, fallback_object_id, fallback_scene_kind, fallback_scale_pose):
    """BODex stores scene_path[0] pointing at the original DGN scene_cfg .npy.
    Use it directly so DRO sees the same scale and object pose."""
    sp = bodex_data.get("scene_path")
    if sp is not None and len(sp) and os.path.exists(str(sp[0])):
        return str(sp[0])
    # Fallback: reconstruct from object_id under DGN_SCENE_CFG_ROOT
    candidate = os.path.join(
        DGN_SCENE_CFG_ROOT, fallback_object_id, fallback_scene_kind,
        f"{fallback_scale_pose}.npy",
    )
    return candidate


def list_bodex_object_dirs(hand_root, object_set, glob_pat):
    """Return [(object_set, object_dir_path), ...] in sorted order. Used by the
    multi-GPU launcher to compute contiguous slices per shard."""
    sets = [object_set] if object_set else sorted(
        d for d in os.listdir(hand_root) if os.path.isdir(os.path.join(hand_root, d))
    )
    out = []
    for s in sets:
        graspdata = os.path.join(hand_root, s, "graspdata")
        if not os.path.isdir(graspdata):
            continue
        for obj_dir in sorted(glob.glob(os.path.join(graspdata, glob_pat))):
            if os.path.isdir(obj_dir):
                out.append((s, obj_dir))
    return out


def iter_bodex_scenes(hand_root, object_set, scene_kind, glob_pat,
                      obj_start=None, obj_end=None):
    """Iterate (object_set, object_id, scale_pose, grasp_npy_path) tuples.

    obj_start / obj_end are inclusive-exclusive indices into the sorted list of
    object directories. They let multiple processes split work without
    overlapping. If both are None, all objects are processed.
    """
    obj_dirs = list_bodex_object_dirs(hand_root, object_set, glob_pat)
    if obj_start is not None or obj_end is not None:
        s = 0 if obj_start is None else int(obj_start)
        e = len(obj_dirs) if obj_end is None else int(obj_end)
        obj_dirs = obj_dirs[s:e]
    for s, obj_dir in obj_dirs:
        object_id = os.path.basename(obj_dir)
        scene_dir = os.path.join(obj_dir, scene_kind)
        if not os.path.isdir(scene_dir):
            continue
        for npy in sorted(glob.glob(os.path.join(scene_dir, "*_grasp.npy"))):
            scale_pose = os.path.basename(npy).replace("_grasp.npy", "")
            yield s, object_id, scale_pose, npy


def dro_predict_q_to_bodex_pose(
    predict_q,                    # (B, 6 + dof_d) DRO order: xyz + intrinsic XYZ euler + finger
    perm_bodex_to_dro,            # list[int]: dro_finger[i] = bodex_finger[perm[i]]
    object_translation_world,     # (3,)
    src_side, dst_side,
):
    """Inverse of bodex_seed_to_dro_initial_q. Returns (B, 7 + dof_b) array."""
    predict_q = np.asarray(predict_q, dtype=np.float64)
    B = predict_q.shape[0]
    base_xyz = predict_q[:, :3] + np.asarray(object_translation_world, dtype=np.float64)
    eulers = predict_q[:, 3:6]
    quat_xyzw = R.from_euler("XYZ", eulers).as_quat()  # (B, 4) in xyzw
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    dro_finger = predict_q[:, 6:]
    n_b = len(perm_bodex_to_dro)
    bodex_finger = np.zeros((B, n_b), dtype=np.float64)
    # perm[i] tells where bodex's i-th joint sits in dro_finger
    for dro_idx, bodex_idx in enumerate(perm_bodex_to_dro):
        bodex_finger[:, bodex_idx] = dro_finger[:, dro_idx]
    out = np.concatenate([base_xyz, quat_wxyz, bodex_finger], axis=1).astype(np.float32)
    return out


def make_three_stage_robot_pose(
    grasp_bodex,
    pregrasp_factor: float = 0.9,
    squeeze_factor: float = 1.05,
):
    """Mimic BODex's pregrasp/grasp/squeeze layout.

    Input  grasp_bodex: (B, 7 + dof_b) -- DRO prediction in BODex format
                                          [tx, ty, tz, qw, qx, qy, qz, joint_q...]
    Output:             (B, 3, 7 + dof_b)

    Stage 0 (pregrasp): same root pose; finger joints scaled by `pregrasp_factor`
                        (default 0.9, i.e. ~10% less curled).
    Stage 1 (grasp):    DRO prediction verbatim.
    Stage 2 (squeeze):  same root pose; finger joints scaled by `squeeze_factor`
                        (default 1.05, i.e. ~5% more curled).

    Multiplicative scaling preserves joint sign (so e.g. negative thumb-rotation
    values move toward 0 in pregrasp regardless of axis convention) and is what
    BODex's stages roughly look like in practice.
    """
    grasp_bodex = np.asarray(grasp_bodex, dtype=np.float32)
    if grasp_bodex.ndim != 2 or grasp_bodex.shape[1] < 8:
        raise ValueError(
            f"expected grasp_bodex shape (B, 7+dof), got {grasp_bodex.shape}"
        )

    pregrasp = grasp_bodex.copy()
    pregrasp[:, 7:] = grasp_bodex[:, 7:] * float(pregrasp_factor)

    grasp = grasp_bodex.copy()

    squeeze = grasp_bodex.copy()
    squeeze[:, 7:] = grasp_bodex[:, 7:] * float(squeeze_factor)

    return np.stack([pregrasp, grasp, squeeze], axis=1)  # (B, 3, 7+dof)


def save_scene_bodex_format(
    out_npy_path, robot_pose, seed_pose, raw_bodex,
):
    """Write a BODex-shaped dict at out_npy_path.

    robot_pose: (B, S, 7 + dof_b) np.float32 -- DRO's predictions in BODex order;
                S is the number of stages (3 = pregrasp/grasp/squeeze).
    seed_pose:  (B, 7 + dof_b) np.float32 -- the seeds we actually fed to DRO.
    raw_bodex:  the original BODex .item() dict; we copy joint_names / world_cfg /
                manip_name / scene_path verbatim so downstream tooling still works.
    """
    robot_pose = np.asarray(robot_pose, dtype=np.float32)
    if robot_pose.ndim == 2:
        # backward-compat: caller passed a single-stage (B, 7+dof) tensor
        robot_pose = robot_pose[:, None, :]
    if robot_pose.ndim != 3:
        raise ValueError(
            f"robot_pose must be (B, S, 7+dof) or (B, 7+dof); got {robot_pose.shape}"
        )
    os.makedirs(os.path.dirname(out_npy_path), exist_ok=True)
    out = {
        "robot_pose": robot_pose[None, ...].astype(np.float32),  # (1, B, S, 7+dof)
        "seed":       seed_pose[None, :, :].astype(np.float32),  # (1, B,    7+dof)
        "joint_names": list(raw_bodex["joint_names"]),
        "world_cfg":   list(raw_bodex.get("world_cfg", [])),
        "manip_name":  list(raw_bodex.get("manip_name", [])),
        "scene_path":  list(raw_bodex.get("scene_path", [])),
        "source": "DRO-Grasp/inference_DGN_bodex_seed.py",
    }
    np.save(out_npy_path, out, allow_pickle=True)


@hydra.main(version_base="1.2", config_path="configs", config_name="validate")
def main(cfg):
    # --- args from cfg (Hydra: pass via `+name=value` on CLI) ---
    bodex_hand = getattr(cfg, "bodex_hand", "sim_xhand/fc_left")
    bodex_input_root = getattr(cfg, "bodex_input_root", BODEX_OUTPUT_ROOT)
    bodex_object_set = getattr(cfg, "bodex_object_set", None)
    bodex_scene_kind = getattr(cfg, "bodex_scene_kind", "tabletop_ur10e")
    bodex_glob = getattr(cfg, "bodex_glob", "*")
    out_dir = getattr(cfg, "out_dir", os.path.join(ROOT_DIR, "dro_bodex_output"))
    pregrasp_factor = float(getattr(cfg, "pregrasp_factor", 0.9))
    squeeze_factor = float(getattr(cfg, "squeeze_factor", 1.05))
    obj_start = getattr(cfg, "obj_start", None)
    obj_end = getattr(cfg, "obj_end", None)
    if obj_start is not None:
        obj_start = int(obj_start)
    if obj_end is not None:
        obj_end = int(obj_end)

    # DRO hand name: explicit override or guess from BODex sim folder
    sim_dir = bodex_hand.split("/")[0]
    side_dir = bodex_hand.split("/")[1] if "/" in bodex_hand else "fc_left"
    src_side = "left" if side_dir.endswith("left") else "right"
    hand_name = getattr(cfg, "hand_name", DEFAULT_HAND_NAME_FOR_BODEX.get(sim_dir))
    if hand_name is None:
        raise SystemExit(
            f"could not infer DRO hand_name from {bodex_hand}; pass +hand_name=..."
        )

    hand_root = os.path.join(bodex_input_root, bodex_hand)
    if not os.path.isdir(hand_root):
        raise SystemExit(f"bodex hand path not found: {hand_root}")

    proposer = GraspPoseProposal(cfg)

    # All DRO right-hand URDFs we care about have `right_*` joint names; we
    # swap left->right (or right->right, no-op) when reading BODex seeds.
    dro_hand = proposer.hand_models.get(hand_name)
    if dro_hand is None:
        # force-load to introspect joint names before the first scene
        from utils.hand_model import create_hand_model
        dro_hand = create_hand_model(hand_name, device=proposer.device)
        proposer.hand_models[hand_name] = dro_hand
    dst_side = "right"  # all DRO URDFs in this repo are right-handed
    dro_joint_param_names_finger = dro_hand.pk_chain.get_joint_parameter_names()[6:]

    hand_tag = bodex_hand.replace("/", "_")
    n_done = 0
    n_skipped = 0
    for object_set, object_id, scale_pose, npy in iter_bodex_scenes(
        hand_root, bodex_object_set, bodex_scene_kind, bodex_glob,
        obj_start=obj_start, obj_end=obj_end,
    ):
        bodex_data = load_bodex_npy(npy)
        bodex_joint_names = list(bodex_data["joint_names"])
        seed_all = bodex_data["seed"][0]   # (N, 7 + dof_b)

        scene_cfg_path = resolve_scene_cfg_path(
            bodex_data, object_id, bodex_scene_kind, scale_pose
        )
        if not os.path.exists(scene_cfg_path):
            print(f"[skip] no scene_cfg for {object_id}/{scale_pose}: {scene_cfg_path}")
            n_skipped += 1
            continue

        info = load_dgn_scene_cfg(scene_cfg_path)
        if not os.path.exists(info["mesh_path"]):
            print(f"[skip] mesh missing: {info['mesh_path']}")
            n_skipped += 1
            continue

        object_translation_world = np.asarray(info["pose"][:3], dtype=np.float64)

        # Build per-batch initial_q. DRO's batch_size (cfg.dataset.batch_size) is
        # how many grasps it predicts per call; we tile/truncate BODex seeds to
        # exactly that count. Pre-build the joint permutation once per scene.
        try:
            perm = build_joint_perm(
                bodex_joint_names, dro_joint_param_names_finger, src_side, dst_side
            )
        except ValueError as e:
            print(f"[skip] joint mapping failed for {object_id}/{scale_pose}: {e}")
            n_skipped += 1
            continue

        bsz = proposer.batch_size
        n_seeds = seed_all.shape[0]
        idx = np.arange(bsz) % n_seeds  # tile if BODex has fewer seeds than bsz
        chosen = seed_all[idx]          # (bsz, 7 + dof_b)

        initial_q_batch = np.zeros((bsz, dro_hand.dof), dtype=np.float32)
        base_xyz = chosen[:, :3] - object_translation_world
        # wxyz -> rotation matrix -> intrinsic XYZ euler
        quat_xyzw = chosen[:, [4, 5, 6, 3]]
        eulers = R.from_quat(quat_xyzw).as_euler("XYZ").astype(np.float32)
        finger_q = chosen[:, 7:][:, perm].astype(np.float32)
        initial_q_batch[:, :3] = base_xyz.astype(np.float32)
        initial_q_batch[:, 3:6] = eulers
        initial_q_batch[:, 6:] = finger_q

        cache_key = info["scene_id"]
        print(f"[run] {hand_tag}  {cache_key}  seeds={n_seeds} -> bsz={bsz}")
        out = proposer.predict_grasp_pose(
            hand_name=hand_name,
            object_name=cache_key,
            object_path=info["mesh_path"],
            object_scale=info["scale"],
            object_pose=info["pose"],
            initial_q=initial_q_batch,
            debug=False,
        )

        # Path mirrors BODex's own layout exactly:
        #   <out_dir>/<bodex_hand>/<obj_set>/graspdata/<obj_id>/<scene>/<scale_pose>_grasp.npy
        out_npy_path = os.path.join(
            out_dir, bodex_hand, object_set, "graspdata",
            object_id, bodex_scene_kind, f"{scale_pose}_grasp.npy",
        )

        predict_q = out["predict_q"].detach().cpu().numpy()  # (B, 6 + dof_d)
        robot_pose_bodex = dro_predict_q_to_bodex_pose(
            predict_q, perm, object_translation_world, src_side, dst_side
        )  # (B, 7 + dof_b)

        robot_pose_3stage = make_three_stage_robot_pose(
            robot_pose_bodex,
            pregrasp_factor=pregrasp_factor,
            squeeze_factor=squeeze_factor,
        )  # (B, 3, 7 + dof_b)

        # Echo back the actual seeds we used (in BODex order, side restored).
        seed_pose_bodex = chosen.astype(np.float32)  # already in BODex coords/order

        save_scene_bodex_format(
            out_npy_path,
            robot_pose=robot_pose_3stage,
            seed_pose=seed_pose_bodex,
            raw_bodex=bodex_data,
        )
        n_done += 1
        if n_done % 10 == 0:
            print(f"[{n_done}] last out: {out_npy_path}")

    print(f"done. predicted {n_done} scenes ({n_skipped} skipped) into {out_dir}")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)
    main()
