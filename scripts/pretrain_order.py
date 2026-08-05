import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import argparse
import warnings
from types import SimpleNamespace
from tqdm import tqdm
import torch

from model.network import create_encoder_network
from data_utils.CMapDataset import create_dataloader
from utils.pretrain_utils import dist2weight, infonce_loss


def main(args):
    encoder = create_encoder_network(emb_dim=512)

    # `--random_init` evaluates the untrained encoder, which is the only meaningful
    # baseline: same robots, same objects, same `get_initial_q()` perturbation, so the
    # number is directly comparable to the trained checkpoints. (Mean order is not
    # comparable across datasets, so the released pretrain_3robots.pth -- trained on
    # barrett / allegro_right / shadowhand -- is not a baseline for other robots.)
    epoch_list = ['random_init'] if args.random_init else args.epoch_list

    for epoch in epoch_list:
        print("****************************************************************")
        print(f"[Epoch {epoch}]")
        if not args.random_init:
            ckpt_path = os.path.join(ROOT_DIR, f'output/{args.pretrain_ckpt}/state_dict/epoch_{epoch}.pth')
            if not os.path.exists(ckpt_path):  # also accept a single-file ckpt/pretrain/*.pth
                released_path = os.path.join(ROOT_DIR, f'ckpt/pretrain/{args.pretrain_ckpt}')
                assert os.path.exists(released_path), f"neither {ckpt_path} nor {released_path} exists"
                ckpt_path = released_path
            encoder.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))

        for robot_name in args.robot_names:
            print(f"Robot: {robot_name}")
            # Fixed seed so every epoch sees the same grasps / `get_initial_q()` perturbations
            # / object point samples; otherwise the epoch-to-epoch differences are dominated
            # by sampling noise. DataLoader derives its worker seeds from the torch RNG.
            torch.manual_seed(args.seed)
            dataloader = create_dataloader(
                SimpleNamespace(**{
                    'batch_size': 1,
                    'robot_names': [robot_name],
                    'debug_object_names': None,
                    'object_pc_type': 'random',
                    'num_workers': 4
                }),
                is_train=True
            )
            # print(len(dataloader))

            orders = []
            for data_idx, data in enumerate(tqdm(dataloader, total=args.data_num)):
                if data_idx == args.data_num:
                    break

                pc_1 = data['robot_pc_initial']
                pc_2 = data['robot_pc_target']

                pc_1 = pc_1 - pc_1.mean(dim=1, keepdims=True)
                pc_2 = pc_2 - pc_2.mean(dim=1, keepdims=True)

                emb_1 = encoder(pc_1).detach()
                emb_2 = encoder(pc_2).detach()

                weight = dist2weight(pc_1, func=lambda x: torch.tanh(10 * x))
                loss, similarity = infonce_loss(
                    emb_1, emb_2, weights=weight, temperature=0.1
                )

                order = (similarity > similarity.diagonal(dim1=-2, dim2=-1).unsqueeze(-1)).sum(-1).float().mean()
                orders.append(order)
                if args.verbose:
                    print("\torder:", order)

            print(f"Epoch: {epoch}, Robot: {robot_name}, Mean Order: {sum(orders) / len(orders)}\n")


if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_ckpt', type=str, default='pretrain_3robots')
    parser.add_argument('--data_num', type=int, default=200)
    parser.add_argument('--epoch_list', type=lambda string: string.split(','),
                        default=['10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    parser.add_argument('--robot_names', type=lambda string: string.split(','),
                        default=['barrett', 'allegro', 'shadowhand'])
    parser.add_argument('--random_init', action='store_true',
                        help='evaluate an untrained encoder (baseline); ignores --epoch_list')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
