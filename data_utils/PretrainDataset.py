import os
import sys
import json
import time
import random
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from utils.hand_model import create_hand_model


class PretrainDataset(Dataset):
    def __init__(self, robot_names: list = None, dataset_name: str = None, train_objects_only: bool = True):
        """
        :param dataset_name: file under data/CMapDataset_filtered/ holding {'metadata': [(q, object, robot), ...]}.
            When None, falls back to the per-robot data/MultiDex_filtered/ layout of the original release.
        :param train_objects_only: keep only grasps on train-split objects, so validate objects stay unseen.
        """
        self.robot_names = robot_names if robot_names is not None \
            else ['barrett', 'allegro', 'shadowhand']

        shared_metadata = None
        if dataset_name is not None:
            dataset_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered', dataset_name)
            shared_metadata = torch.load(dataset_path)['metadata']
            if train_objects_only:
                split_path = os.path.join(ROOT_DIR, 'data/CMapDataset_filtered/split_train_validate_objects.json')
                train_objects = set(json.load(open(split_path))['train'])
                shared_metadata = [m for m in shared_metadata if m[1] in train_objects]

        self.dataset_len = 0
        self.robot_len = {}
        self.hands = {}
        self.dofs = []
        self.dataset = {}
        for robot_name in self.robot_names:
            self.hands[robot_name] = create_hand_model(robot_name, torch.device('cpu'))
            self.dofs.append(len(self.hands[robot_name].pk_chain.get_joint_parameter_names()))
            self.dataset[robot_name] = []

            if shared_metadata is not None:
                metadata = [m for m in shared_metadata if m[2] == robot_name]
                assert len(metadata) > 0, f"no grasp found for '{robot_name}' in {dataset_name}"
            else:
                dataset_path = os.path.join(ROOT_DIR, f'data/MultiDex_filtered/{robot_name}/{robot_name}.pt')
                dataset = torch.load(dataset_path)
                metadata = dataset['metadata']
            self.dataset[robot_name].extend(metadata)
            self.dataset_len += len(metadata)
            self.robot_len[robot_name] = len(metadata)

    def __getitem__(self, index):
        robot_name = random.choices(self.robot_names, weights=self.dofs, k=1)[0]

        hand = self.hands[robot_name]
        dataset = self.dataset[robot_name]
        target_q, _, _ = random.choice(dataset)

        robot_pc_1 = hand.get_transformed_links_pc(target_q)[:, :3]
        initial_q = hand.get_initial_q(target_q)
        robot_pc_2 = hand.get_transformed_links_pc(initial_q)[:, :3]

        return {
            'robot_pc_1': robot_pc_1,
            'robot_pc_2': robot_pc_2,
        }

    def __len__(self):
        return self.dataset_len


def create_dataloader(cfg):
    dataset = PretrainDataset(cfg.robot_names, dataset_name=cfg.get('dataset_name', None))
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=True
    )
    return dataloader
