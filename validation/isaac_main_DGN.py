"""IsaacGym validation entry point for DGN grasps produced by DRO.

Differences from validation/isaac_main.py:
- Loads the DGN object URDF directly (data/object/DGN_2k_origin/processed_data/
  <object_id>/urdf/coacd.urdf) instead of the data/data_urdf/object/<set>/<id>/
  layout, so we don't need a duplicate URDF tree for the 2k DGN bottles.
- Bakes the per-scene per-axis object scale into a temp URDF on the fly. The
  temp URDF rewrites every <mesh ... scale="..."/> attribute and absolutizes
  mesh filenames so IsaacGym can load it from outside the original directory.
- Caller is expected to have already expressed the hand pose (q_batch) in the
  object's NATIVE mesh frame (i.e. with the scene-cfg rotation undone), so we
  can keep the IsaacValidator default of object-at-origin/identity.

Invoked as a subprocess by validate_DGN.py to avoid Isaac Gym GPU memory leaks.
"""

from isaacgym import gymapi  # noqa: F401  IsaacGym must be imported before PyTorch
from isaacgym import gymtorch  # noqa: F401

import os
import sys
import re
import json
import shutil
import tempfile
import argparse
import warnings
from termcolor import cprint

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from validation.isaac_validator import IsaacValidator
from utils.hand_model import create_hand_model
from utils.rotation import q_rot6d_to_q_euler


_MESH_TAG_RE = re.compile(
    r'<mesh\s+filename="([^"]+)"(?:\s+scale="[^"]*")?\s*/>'
)


def write_scaled_dgn_urdf(orig_urdf: str, scale_xyz, out_path: str) -> str:
    """Copy a DGN coacd.urdf into out_path, rewriting every <mesh .../> tag with
    the requested per-axis scale and absolute mesh paths so IsaacGym can load
    the URDF from anywhere on disk."""
    orig_urdf = os.path.abspath(orig_urdf)
    base_dir = os.path.dirname(orig_urdf)
    sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])

    with open(orig_urdf, "r") as f:
        text = f.read()

    def repl(m):
        filename = m.group(1)
        if not os.path.isabs(filename):
            filename = os.path.normpath(os.path.join(base_dir, filename))
        return f'<mesh filename="{filename}" scale="{sx} {sy} {sz}"/>'

    text = _MESH_TAG_RE.sub(repl, text)
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def isaac_main_dgn(
    robot_name: str,
    object_urdf: str,
    scale_xyz,
    q_batch: torch.Tensor,
    gpu: int = 0,
    use_gui: bool = False,
):
    """Run a single DGN scene through IsaacGym and return (success, q_isaac).

    q_batch is expected to be in the object's NATIVE mesh frame (no scene
    rotation applied) so we can keep the validator's default object pose
    (origin, identity).
    """
    if use_gui:
        gpu = 0

    data_urdf_path = os.path.join(ROOT_DIR, "data/data_urdf")
    urdf_assets_meta = json.load(
        open(os.path.join(data_urdf_path, "robot/urdf_assets_meta.json"))
    )
    robot_urdf_path = urdf_assets_meta["urdf_path"][robot_name]

    hand = create_hand_model(robot_name)
    joint_orders = hand.get_joint_orders()
    if q_batch.shape[-1] != len(joint_orders):
        q_batch = q_rot6d_to_q_euler(q_batch)
    batch_size = q_batch.shape[0]

    tmp_dir = tempfile.mkdtemp(prefix="dro_dgn_urdf_")
    try:
        scaled_urdf = os.path.join(tmp_dir, "object_scaled.urdf")
        write_scaled_dgn_urdf(object_urdf, scale_xyz, scaled_urdf)

        simulator = IsaacValidator(
            robot_name=robot_name,
            joint_orders=joint_orders,
            batch_size=batch_size,
            gpu=gpu,
            is_filter=False,
            use_gui=use_gui,
        )
        print("[Isaac/DGN] IsaacValidator is created.")

        simulator.set_asset(
            robot_path=os.path.join(data_urdf_path, "robot"),
            robot_file=robot_urdf_path[len("data/data_urdf/robot/"):],
            object_path=tmp_dir,
            object_file=os.path.basename(scaled_urdf),
        )
        simulator.create_envs()
        print("[Isaac/DGN] IsaacValidator preparation is done.")

        simulator.set_actor_pose_dof(q_batch.to(torch.device("cpu")))
        success, q_isaac = simulator.run_sim()
        simulator.destroy()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return success, q_isaac


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    p = argparse.ArgumentParser()
    p.add_argument("--robot_name", required=True)
    p.add_argument("--object_urdf", required=True,
                   help="absolute path to coacd.urdf for this DGN object")
    p.add_argument("--object_tag", required=True,
                   help="logging tag, typically <object_id>/<scale_pose>")
    p.add_argument("--scale_x", type=float, required=True)
    p.add_argument("--scale_y", type=float, required=True)
    p.add_argument("--scale_z", type=float, required=True)
    p.add_argument("--q_file", required=True)
    p.add_argument("--ret_file", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--use_gui", action="store_true")
    args = p.parse_args()

    print(f"GPU: {args.gpu}")
    q_batch = torch.load(args.q_file, map_location="cpu")
    success, q_isaac = isaac_main_dgn(
        robot_name=args.robot_name,
        object_urdf=args.object_urdf,
        scale_xyz=(args.scale_x, args.scale_y, args.scale_z),
        q_batch=q_batch,
        gpu=args.gpu,
        use_gui=args.use_gui,
    )
    success_num = success.sum().item()
    cprint(
        f"[{args.robot_name}/{args.object_tag}] Result: {success_num}/{q_batch.shape[0]}",
        "green",
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.ret_file)), exist_ok=True)
    torch.save({"success": success, "q_isaac": q_isaac}, args.ret_file)
