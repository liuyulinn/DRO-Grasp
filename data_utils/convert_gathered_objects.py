"""
Convert the YCB meshes used by `~/DexCom/assets/gathered_states` into the format
CMapDataset expects:

    data/data_urdf/object/ycb/<object_name>/<object_name>.stl   (scaled mesh)
    data/PointCloud/object/ycb/<object_name>.pt                 (num_points, 6) xyz + normal

The gathered_states grasps were generated against `<obj>/processed_mesh/normalized.obj`
uniformly scaled by `obj_scale` (0.1 for every chunk), so the same scale is baked in here.
The resulting mesh is the canonical object frame that grasp q values are expressed in.
"""

import os
import sys
import argparse
import numpy as np
import torch
import trimesh

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)


def convert_objects(args):
    mesh_out_dir = os.path.join(ROOT_DIR, 'data/data_urdf/object', args.dataset_type)
    pc_out_dir = os.path.join(ROOT_DIR, 'data/PointCloud/object', args.dataset_type)
    os.makedirs(mesh_out_dir, exist_ok=True)
    os.makedirs(pc_out_dir, exist_ok=True)

    object_names = sorted(
        d for d in os.listdir(args.source_dir)
        if os.path.isdir(os.path.join(args.source_dir, d))
    )
    print(f"Found {len(object_names)} objects in {args.source_dir}")

    for object_name in object_names:
        source_path = os.path.join(args.source_dir, object_name, 'processed_mesh', 'normalized.obj')
        if not os.path.exists(source_path):
            print(f"  [skip] {object_name}: no processed_mesh/normalized.obj")
            continue

        mesh = trimesh.load(source_path, force='mesh')
        mesh.apply_scale(args.scale)

        object_dir = os.path.join(mesh_out_dir, object_name)
        os.makedirs(object_dir, exist_ok=True)
        mesh.export(os.path.join(object_dir, f'{object_name}.stl'))

        # (num_points, 6): surface point xyz + face normal, used by loss_depth
        object_pc, face_indices = mesh.sample(args.num_points, return_index=True)
        object_pc = torch.tensor(object_pc, dtype=torch.float32)
        normals = torch.tensor(mesh.face_normals[face_indices], dtype=torch.float32)
        torch.save(torch.cat([object_pc, normals], dim=-1), os.path.join(pc_out_dir, f'{object_name}.pt'))

        print(f"  {object_name:26s} extents {np.round(mesh.extents, 4)} "
              f"bbox_center {np.round(mesh.bounds.mean(0), 5)}")

    print("\nConverting object meshes & point clouds finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=os.path.expanduser('~/DexCom/assets/misc/ycb_new'), type=str)
    parser.add_argument('--dataset_type', default='ycb', type=str)
    parser.add_argument('--scale', default=0.1, type=float, help='obj_scale used when generating the grasps')
    parser.add_argument('--num_points', default=512, type=int)
    args = parser.parse_args()

    convert_objects(args)
