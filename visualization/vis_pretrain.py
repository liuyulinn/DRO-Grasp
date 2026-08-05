"""
Viser viewer for the pretrained encoder's point-wise correspondence.

The initial point cloud is rainbow-coloured along y; every point of the target cloud
takes the colour of the initial point the encoder matched it to. A good encoder gives
a smooth rainbow on the target too -- colour noise means the correspondence is wrong.

    # a checkpoint from a training run, i.e. output/<name>/state_dict/epoch_<N>.pth
    python visualization/vis_pretrain.py --pretrain_ckpt pretrain_3lefthands --epoch 5

    # a single-file release, i.e. ckpt/pretrain/<file>
    python visualization/vis_pretrain.py --pretrain_ckpt pretrain_3robots.pth --robot_names shadowhand

    # untrained encoder, as a baseline for what "bad" looks like
    python visualization/vis_pretrain.py --random_init

Use the GUI to switch robot / resample the grasp / step through epochs without restarting.
"""

import os
import sys
import time
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import matplotlib.pyplot as plt
import torch
import viser

from model.network import create_encoder_network
from data_utils.CMapDataset import CMapDataset
from utils.pretrain_utils import dist2weight, infonce_loss
from utils.hand_model import create_hand_model


def resolve_ckpt_path(pretrain_ckpt, epoch):
    """Accept both `output/<run>/state_dict/epoch_<N>.pth` and `ckpt/pretrain/<file>`."""
    run_path = os.path.join(ROOT_DIR, f'output/{pretrain_ckpt}/state_dict/epoch_{epoch}.pth')
    if os.path.exists(run_path):
        return run_path
    released_path = os.path.join(ROOT_DIR, f'ckpt/pretrain/{pretrain_ckpt}')
    if os.path.exists(released_path):
        return released_path
    raise FileNotFoundError(f"neither {run_path} nor {released_path} exists")


def available_epochs(pretrain_ckpt):
    """Epochs saved so far, so the GUI can follow a run that is still training."""
    state_dir = os.path.join(ROOT_DIR, f'output/{pretrain_ckpt}/state_dict')
    if not os.path.isdir(state_dir):
        return []
    epochs = [int(f[len('epoch_'):-len('.pth')]) for f in os.listdir(state_dir)
              if f.startswith('epoch_') and f.endswith('.pth')]
    return sorted(epochs)


def main(args):
    torch.manual_seed(args.seed)
    encoder = create_encoder_network(emb_dim=512)  # random weights until a ckpt is loaded
    dataset_cache = {}

    def get_dataset(robot_name):
        if robot_name not in dataset_cache:
            print(f"Loading dataset for '{robot_name}'...")
            dataset_cache[robot_name] = CMapDataset(
                batch_size=1,
                robot_names=[robot_name],
                is_train=True,
                debug_object_names=None,
                object_pc_type=args.object_pc_type
            )
        return dataset_cache[robot_name]

    hand_cache = {}

    def get_hand(robot_name):
        if robot_name not in hand_cache:
            hand_cache[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
        return hand_cache[robot_name]

    server = viser.ViserServer(host=args.host, port=args.port)

    robot_dropdown = server.gui.add_dropdown(
        'robot', options=args.robot_names, initial_value=args.robot_names[0])
    epochs = available_epochs(args.pretrain_ckpt)
    epoch_dropdown = server.gui.add_dropdown(
        'epoch',
        options=[str(e) for e in epochs] if epochs else [str(args.epoch)],
        initial_value=str(args.epoch if args.epoch in epochs or not epochs else epochs[-1]),
        disabled=args.random_init
    )
    refresh_button = server.gui.add_button('refresh epoch list')
    resample_button = server.gui.add_button('resample grasp')
    info_text = server.gui.add_text('info', initial_value='', disabled=True)

    state = {'ckpt': None}

    def on_update(_=None):
        robot_name = robot_dropdown.value

        if args.random_init:
            state['ckpt'] = 'random_init'
        else:
            ckpt_path = resolve_ckpt_path(args.pretrain_ckpt, epoch_dropdown.value)
            if state['ckpt'] != ckpt_path:
                print(f"Loading {ckpt_path}")
                encoder.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
                state['ckpt'] = ckpt_path
        encoder.eval()

        data = get_dataset(robot_name)[0]
        q_1 = data['initial_q'][0]
        q_2 = data['target_q'][0].clone()
        pc_1 = data['robot_pc_initial']
        pc_2 = data['robot_pc_target']

        emb_1 = encoder(pc_1 - pc_1.mean(dim=1, keepdims=True)).detach()
        emb_2 = encoder(pc_2 - pc_2.mean(dim=1, keepdims=True)).detach()

        weight = dist2weight(pc_1 - pc_1.mean(dim=1, keepdims=True), func=lambda x: torch.tanh(10 * x))
        loss, similarity = infonce_loss(emb_1, emb_2, weights=weight, temperature=0.1)
        match_idx = torch.argmax(similarity[0], dim=0)
        order = (similarity > similarity.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)).sum(-1).float().mean()

        # offset for clearer visualization result
        offset = torch.tensor([0, 0.3, 0])
        vis_pc_1 = pc_1[0]
        vis_pc_2 = pc_2[0] + offset
        q_2[:3] += offset

        y_values = vis_pc_1[:, 1]
        y_normalized = (y_values - y_values.min()) / (y_values.max() - y_values.min())
        cmap = plt.get_cmap('rainbow')
        initial_colors = cmap(y_normalized)[:, :3]
        target_colors = initial_colors[match_idx]

        server.scene.add_point_cloud(
            'initial pc', vis_pc_1[:, :3].numpy(),
            point_size=0.002, point_shape='circle', colors=initial_colors)
        server.scene.add_point_cloud(
            'target pc', vis_pc_2[:, :3].numpy(),
            point_size=0.002, point_shape='circle', colors=target_colors)

        hand = get_hand(robot_name)
        for name, q in (('robot_initial', q_1), ('robot_target', q_2)):
            robot_trimesh = hand.get_trimesh_q(q)['visual']
            server.scene.add_mesh_simple(
                name, robot_trimesh.vertices, robot_trimesh.faces, color=(102, 192, 255), opacity=0.2)

        info_text.value = (f'{data["object_name"][0]}  '
                           f'order {order.item():.1f}/{similarity.shape[-1]}  '
                           f'loss {loss.item():.3f}')
        print(f'[{robot_name}] {state["ckpt"]}: {info_text.value}')

    def on_refresh(_=None):
        epochs = available_epochs(args.pretrain_ckpt)
        if epochs:
            epoch_dropdown.options = [str(e) for e in epochs]

    for handle in (robot_dropdown, epoch_dropdown):
        handle.on_update(on_update)
    resample_button.on_click(on_update)
    refresh_button.on_click(on_refresh)
    on_update()

    print(f"\nViser server running at http://{args.host}:{args.port} -- Ctrl-C to quit.")
    while True:
        time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_ckpt', type=str, default='pretrain_3lefthands',
                        help='an output/ run name, or a file name under ckpt/pretrain/')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--robot_names', type=lambda s: s.split(','),
                        default=['allegro_left', 'xhand_left', 'fixsharpa_left'])
    parser.add_argument('--object_pc_type', type=str, default='random', choices=['fixed', 'random', 'partial'])
    parser.add_argument('--random_init', action='store_true',
                        help='visualize an untrained encoder (baseline); ignores --pretrain_ckpt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--host', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=8080, type=int)
    args = parser.parse_args()

    main(args)
