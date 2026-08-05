"""
Convert `~/DexCom/assets/gathered_states/<hand>_multipose/` grasps into the
CMapDataset format:

    data/CMapDataset_filtered/cmap_dataset.pt = {
        'info': {<robot_name>: {'robot_name', 'num_total', 'num_upper_object', 'num_per_object'}},
        'metadata': [(q, '<dataset_type>+<object_name>', <robot_name>), ...]
    }

Two representation changes are needed (see plan.md §2):

1. Frame. `grasp_qpos` is stored in the world/tabletop frame with the object pose kept
   separately in `obj_pose`; CMapDataset expects q relative to the canonical (centered)
   object mesh. So the hand base is transformed by `inv(T_obj)`.
2. Rotation. `grasp_qpos` uses [xyz, wxyz quaternion]; CMapDataset uses
   [xyz, intrinsic-XYZ euler] (`utils/rotation.py`).

Joint values need NO permutation: for all three hands `npz['joint_names']` is
element-wise identical to `pk_chain.get_joint_parameter_names()[6:]`, which the script
asserts per chunk. Do not copy the permutation in
`~/DexCom/utils/grasp_data/filtered_bodex.py` -- that one reorders into DexCom's
anatomical joint order and would silently swap allegro_left's index and ring fingers.
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


def pose_to_matrix(pose):
    """[xyz(3), wxyz quat(4)] -> (4, 4) homogeneous transform."""
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()  # wxyz -> xyzw
    matrix[:3, 3] = pose[:3]
    return matrix


def convert_chunk(npz_path, joint_names):
    """
    Read one chunk and return the successful grasps as q in the object frame.

    :return: (M, 6 + DOF) float32 array, [xyz, intrinsic-XYZ euler, joint_q]
    """
    data = np.load(npz_path, allow_pickle=True)

    chunk_joint_names = data['joint_names'].tolist()
    assert chunk_joint_names == joint_names, (
        f"joint order mismatch in {npz_path}:\n  npz: {chunk_joint_names}\n  urdf: {joint_names}"
    )

    success = data['success'].astype(bool)
    if not success.any():  # some chunks contain no successful grasp at all
        return None

    grasp_qpos = data['grasp_qpos'][success].astype(np.float64)  # (M, 7 + DOF)
    if 'obj_pose' in data.files:
        obj_pose = data['obj_pose'][success].astype(np.float64)  # (M, 7)
    else:
        obj_pose = np.zeros((grasp_qpos.shape[0], 7))
        obj_pose[:, 3] = 1.0

    # obj_pose is constant within a chunk, so inv(T_obj) is computed once
    assert np.unique(obj_pose, axis=0).shape[0] == 1, f"obj_pose varies within {npz_path}"
    obj_matrix_inv = np.linalg.inv(pose_to_matrix(obj_pose[0]))

    hand_matrix = np.tile(np.eye(4), (grasp_qpos.shape[0], 1, 1))
    hand_matrix[:, :3, :3] = Rotation.from_quat(grasp_qpos[:, [4, 5, 6, 3]]).as_matrix()
    hand_matrix[:, :3, 3] = grasp_qpos[:, :3]

    new_matrix = obj_matrix_inv[None] @ hand_matrix
    translation = new_matrix[:, :3, 3]
    euler = Rotation.from_matrix(new_matrix[:, :3, :3]).as_euler('XYZ')

    q = np.concatenate([translation, euler, grasp_qpos[:, 7:]], axis=-1)
    return q.astype(np.float32)


def convert_hand(robot_name, source_dir, dataset_type, num_per_object, verbose=True):
    """Convert every chunk of one hand. Returns (metadata list, info dict)."""
    hand = create_hand_model(robot_name, torch.device('cpu'))
    joint_names = hand.pk_chain.get_joint_parameter_names()[6:]

    object_dirs = sorted(
        d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
    )

    metadata = []
    num_per_object_dict = {}
    num_empty_chunks = 0
    for object_dir in tqdm(object_dirs, desc=robot_name, disable=not verbose):
        # directories are named '<object_name>.obj'
        object_name = object_dir[:-4] if object_dir.endswith('.obj') else object_dir
        full_object_name = f'{dataset_type}+{object_name}'

        mesh_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{dataset_type}/{object_name}/{object_name}.stl')
        assert os.path.exists(mesh_path), \
            f"missing object mesh {mesh_path}, run data_utils/convert_gathered_objects.py first"

        q_list = []
        for npz_path in sorted(glob.glob(os.path.join(source_dir, object_dir, '*', '*.npz'))):
            q = convert_chunk(npz_path, joint_names)
            if q is None:
                num_empty_chunks += 1
                continue
            q_list.append(q)

        if not q_list:
            print(f"  [warn] {robot_name}/{object_name}: no successful grasp, skipped")
            continue

        q_all = torch.tensor(np.concatenate(q_list, axis=0))
        if num_per_object is not None and q_all.shape[0] > num_per_object:
            indices = torch.randperm(q_all.shape[0])[:num_per_object]
            q_all = q_all[indices]

        for idx in range(q_all.shape[0]):
            metadata.append((q_all[idx], full_object_name, robot_name))
        num_per_object_dict[full_object_name] = q_all.shape[0]

    info = {
        'robot_name': robot_name,
        'num_total': sum(num_per_object_dict.values()),
        'num_upper_object': max(num_per_object_dict.values()) if num_per_object_dict else 0,
        'num_per_object': num_per_object_dict
    }
    if verbose:
        print(f"  {robot_name}: {info['num_total']} grasps over {len(num_per_object_dict)} objects "
              f"({num_empty_chunks} all-failed chunks skipped)")
    return metadata, info


def main(args):
    torch.manual_seed(args.seed)

    info = {}
    metadata = []
    for robot_name in args.robot_names:
        source_dir = os.path.join(args.source_dir, f'{robot_name}_{args.suffix}')
        assert os.path.isdir(source_dir), f"{source_dir} does not exist"
        hand_metadata, hand_info = convert_hand(
            robot_name, source_dir, args.dataset_type, args.num_per_object
        )
        metadata += hand_metadata
        info[robot_name] = hand_info

    output_dir = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_name)
    torch.save({'info': info, 'metadata': metadata}, output_path)
    print(f"\nSaved {len(metadata)} grasps to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=os.path.expanduser('~/DexCom/assets/gathered_states'), type=str)
    parser.add_argument('--robot_names', default=['allegro_left', 'xhand_left', 'fixsharpa_left'],
                        type=lambda s: s.split(','))
    parser.add_argument('--suffix', default='multipose', type=str)
    parser.add_argument('--dataset_type', default='ycb', type=str)
    parser.add_argument('--num_per_object', default=None, type=int,
                        help='randomly subsample to at most this many grasps per object')
    parser.add_argument('--output_name', default='cmap_dataset.pt', type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    main(args)
