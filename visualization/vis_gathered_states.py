"""
Viser viewer to sanity-check the gathered_states -> CMapDataset conversion.

Reads grasps straight from `~/DexCom/assets/gathered_states` through
`data_utils/convert_gathered_states.convert_chunk()`, so it can be used *before*
`cmap_dataset.pt` is built. Pass `--source dataset` to inspect the saved
`cmap_dataset.pt` instead.

    python visualization/vis_gathered_states.py
    python visualization/vis_gathered_states.py --source dataset --port 8081

Then open the printed URL. The `min dist` readout is the distance from the closest
robot point-cloud point to the object surface: it should be near 0 for a real grasp
(fingers touching), and clearly negative-looking penetration or tens of millimetres
of clearance means the conversion is off.
"""

import os
import sys
import glob
import time
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import torch
import trimesh
import viser
from scipy.spatial import cKDTree

from utils.hand_model import create_hand_model
from data_utils.convert_gathered_states import convert_chunk


def main(args):
    hand_cache = {}
    object_cache = {}
    grasp_cache = {}

    def get_hand(robot_name):
        if robot_name not in hand_cache:
            print(f"Loading hand model '{robot_name}'...")
            hand_cache[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
        return hand_cache[robot_name]

    def get_object(object_name):
        """Return (trimesh, cKDTree over densely sampled surface)."""
        if object_name not in object_cache:
            mesh_path = os.path.join(
                ROOT_DIR, f'data/data_urdf/object/{args.dataset_type}/{object_name}/{object_name}.stl')
            mesh = trimesh.load_mesh(mesh_path)
            surface, _ = trimesh.sample.sample_surface(mesh, args.num_surface_points)
            object_cache[object_name] = (mesh, cKDTree(surface))
        return object_cache[object_name]

    if args.source == 'dataset':
        dataset_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered', args.dataset_name)
        print(f"Loading {dataset_path}...")
        metadata = torch.load(dataset_path, map_location='cpu')['metadata']
        robot_names = sorted({m[2] for m in metadata})
        object_names = sorted({m[1].split('+')[1] for m in metadata})

        def get_grasps(robot_name, object_name):
            key = (robot_name, object_name)
            if key not in grasp_cache:
                full_name = f'{args.dataset_type}+{object_name}'
                q = [m[0] for m in metadata if m[1] == full_name and m[2] == robot_name]
                grasp_cache[key] = torch.stack(q) if q else None
            return grasp_cache[key]
    else:
        robot_names = args.robot_names
        object_names = sorted(
            d[:-4] if d.endswith('.obj') else d
            for d in os.listdir(os.path.join(args.source_dir, f'{robot_names[0]}_{args.suffix}'))
            if os.path.isdir(os.path.join(args.source_dir, f'{robot_names[0]}_{args.suffix}', d))
        )

        def get_grasps(robot_name, object_name):
            key = (robot_name, object_name)
            if key not in grasp_cache:
                joint_names = get_hand(robot_name).pk_chain.get_joint_parameter_names()[6:]
                pattern = os.path.join(
                    args.source_dir, f'{robot_name}_{args.suffix}', f'{object_name}.obj', '*', '*.npz')
                q_list = []
                for npz_path in sorted(glob.glob(pattern)):
                    q = convert_chunk(npz_path, joint_names)
                    if q is not None:
                        q_list.append(torch.tensor(q))
                    if q_list and sum(x.shape[0] for x in q_list) >= args.max_grasps:
                        break
                grasp_cache[key] = torch.cat(q_list)[:args.max_grasps] if q_list else None
            return grasp_cache[key]

    print(f"robots:  {robot_names}")
    print(f"objects: {len(object_names)}")

    server = viser.ViserServer(host=args.host, port=args.port)

    robot_dropdown = server.gui.add_dropdown('robot', options=robot_names, initial_value=robot_names[0])
    object_dropdown = server.gui.add_dropdown('object', options=object_names, initial_value=object_names[0])
    grasp_slider = server.gui.add_slider('grasp', min=0, max=args.max_grasps - 1, step=1, initial_value=0)
    show_pc_checkbox = server.gui.add_checkbox('show robot point cloud', initial_value=False)
    info_text = server.gui.add_text('info', initial_value='', disabled=True)

    def on_update(_=None):
        robot_name = robot_dropdown.value
        object_name = object_dropdown.value

        q_all = get_grasps(robot_name, object_name)
        if q_all is None or q_all.shape[0] == 0:
            info_text.value = 'no successful grasp'
            return
        grasp_idx = grasp_slider.value % q_all.shape[0]
        q = q_all[grasp_idx]

        object_trimesh, surface_tree = get_object(object_name)
        server.scene.add_mesh_simple(
            'object', object_trimesh.vertices, object_trimesh.faces, color=(239, 132, 167), opacity=1)

        hand = get_hand(robot_name)
        robot_trimesh = hand.get_trimesh_q(q)['visual']
        server.scene.add_mesh_simple(
            'robot', robot_trimesh.vertices, robot_trimesh.faces, color=(102, 192, 255), opacity=0.8)

        robot_pc = hand.get_transformed_links_pc(q)[:, :3].numpy()
        if show_pc_checkbox.value:
            server.scene.add_point_cloud(
                'robot_pc', robot_pc, point_size=0.0015, point_shape='circle', colors=(0, 0, 200))
        else:
            server.scene.add_point_cloud('robot_pc', np.zeros((1, 3)), point_size=0.0, colors=(0, 0, 0))

        dist, _ = surface_tree.query(robot_pc)
        info_text.value = (f'{grasp_idx + 1}/{q_all.shape[0]}  '
                           f'min {dist.min() * 1000:.1f}mm  '
                           f'p1 {np.percentile(dist, 1) * 1000:.1f}mm')
        print(f'[{robot_name}] {object_name} grasp {grasp_idx}: '
              f'{q_all.shape[0]} grasps, min dist {dist.min() * 1000:.2f} mm')

    for handle in (robot_dropdown, object_dropdown, grasp_slider, show_pc_checkbox):
        handle.on_update(on_update)
    on_update()

    print(f"\nViser server running at http://{args.host}:{args.port} -- Ctrl-C to quit.")
    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='gathered', choices=['gathered', 'dataset'],
                        help="'gathered' reads npz directly, 'dataset' reads cmap_dataset.pt")
    parser.add_argument('--source_dir', default=os.path.expanduser('~/DexCom/assets/gathered_states'), type=str)
    parser.add_argument('--robot_names', default=['allegro_left', 'xhand_left', 'fixsharpa_left'],
                        type=lambda s: s.split(','))
    parser.add_argument('--suffix', default='multipose', type=str)
    parser.add_argument('--dataset_type', default='ycb', type=str)
    parser.add_argument('--dataset_name', default='cmap_dataset.pt', type=str)
    parser.add_argument('--max_grasps', default=200, type=int, help='grasps loaded per (robot, object)')
    parser.add_argument('--num_surface_points', default=120000, type=int)
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=8080, type=int)
    args = parser.parse_args()

    main(args)
