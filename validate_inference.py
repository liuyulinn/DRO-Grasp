import os
import sys
import warnings
import hydra
from termcolor import cprint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference_dro import GraspPoseProposal
from validation.validate_utils import validate_isaac


@hydra.main(version_base="1.2", config_path="configs", config_name="validate")
def main(cfg):
    proposer = GraspPoseProposal(cfg)
    hand_name = "allegro"

    DGN = "/home/yulin/workspace/DexCom/assets/misc/DGN_2k_origin/processed_data"
    DGN_SCALE = 0.1
    ids = ["core_bottle_1a7ba1f4c892e2da30711cdbdbc73924"]
    objects = [(f"dgn+{i}", f"{DGN}/{i}/mesh/normalized.obj", DGN_SCALE) for i in ids]

    total_succ, total_num = 0, 0
    for object_name, object_path, scale in objects:
        out = proposer.predict_grasp_pose(hand_name, object_name, object_path, scale=scale)
        success, _ = validate_isaac(hand_name, object_name, out["predict_q"], gpu=cfg.gpu)
        n_succ = int(success.sum().item()) if success is not None else -1
        n = out["predict_q"].shape[0]
        cprint(f"[{hand_name}/{object_name}] {n_succ}/{n} ({n_succ / n * 100:.2f}%)", "green")
        total_succ += n_succ
        total_num += n

    cprint(f"[TOTAL] {total_succ}/{total_num} ({total_succ / total_num * 100:.2f}%)", "yellow")


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    main()

