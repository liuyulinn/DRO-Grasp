import os
import sys
import torch
import trimesh
import numpy as np
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
# --- Assume these utility functions are in your project structure ---
# You would import these from your existing project files.
from model.network import create_network
from utils.hand_model import create_hand_model
from utils.multilateration import multilateration
from utils.se3_transform import compute_link_pose
from utils.optimization import process_transform, create_problem, optimization
import warnings

from utils_dexwm.wis3d_new import Wis3D, SAPIENKinematicsModelStandalone
import hydra

# --- Global variables for models to avoid reloading on every call ---
# This is more efficient than loading them inside the function every time.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NETWORK = None
HAND_MODELS = {}

checkpoint_path = (
    f"{ROOT_DIR}/ckpt/model/model_3robots.pth"  # IMPORTANT: Change this path
)


# how to use this hydra here ?
# @hydra.main(version_base="1.2", config_path="configs", config_name="validate")
class GraspPoseProposal:
    def __init__(self, cfg):  # , hand_names: str):
        device = torch.device(f"cuda:{cfg.gpu}")
        self.device = device
        self.cfg = cfg

        self.batch_size = cfg.dataset.batch_size
        # self.hand_names = hand_names
        self.num_points = 512
        # import pdb; pdb.set_trace()
        self.object_pc_type = cfg.dataset.object_pc_type

        network = create_network(cfg.model, mode="validate").to(device)
        network.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
            )
        )
        network.eval()

        self.network = network
        self.hand_models = {}
        # for name in self.hand_names:
        #     self.hand_models[name] = create_hand_model(name, device=device)
        self.object_pcs = {}

    def prepare_data(
        self, hand_name: str, object_name: str, object_path: str
    ):  # , N: int):
        if hand_name not in self.hand_models:
            self.hand_models[hand_name] = create_hand_model(
                hand_name, device=self.device
            )
        hand = self.hand_models[hand_name]

        # num_points = 512  # Number of points to sample from object and hand
        # self.object_pcs = {}
        # if self.object_pcs

        if object_name in self.object_pcs:
            object_pc = self.object_pcs[object_name]
        else:
            if self.object_pc_type != "fixed":
                # for object_name in self.object_names:
                # name = object_name.split("+")
                # mesh_path = os.path.join(object_path)
                object_pcld_path = object_path.replace(".obj", "_pcd.ply")
                if not os.path.exists(object_pcld_path):
                    mesh = trimesh.load_mesh(object_path)
                    object_pc, _ = mesh.sample(65536, return_index=True)
                    indices = torch.randperm(65536)[: self.num_points]
                    object_pc = object_pc[indices]
                    self.object_pcs[object_name] = torch.tensor(
                        object_pc, dtype=torch.float32
                    )

                    torch.save(self.object_pcs[object_name], object_pcld_path)
                else:
                    self.object_pcs[object_name] = torch.load(
                        object_pcld_path, weights_only=False
                    )
            else:
                print("!!! Using fixed object pcs !!!")

        initial_q_batch = torch.zeros([self.batch_size, hand.dof], dtype=torch.float32)
        robot_pc_batch = torch.zeros(
            [self.batch_size, self.num_points, 3], dtype=torch.float32
        )
        object_pc_batch = torch.zeros(
            [self.batch_size, self.num_points, 3], dtype=torch.float32
        )

        for batch_idx in range(self.batch_size):
            initial_q = hand.get_initial_q()
            robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3]

            if self.object_pc_type == "partial":
                indices = torch.randperm(65536)[: self.num_points * 2]
                object_pc = self.object_pcs[object_name][indices]
                direction = torch.randn(3)
                direction = direction / torch.norm(direction)
                proj = object_pc @ direction
                _, indices = torch.sort(proj)
                indices = indices[self.num_points :]
                object_pc = object_pc[indices]
            else:
                object_pc = self.object_pcs[object_name]
                # name = object_name.split("+")
                # object_path = os.path.join(
                #     ROOT_DIR, f"data/PointCloud/object/{name[0]}/{name[1]}.pt"
                # )
                # object_pc = torch.load(object_path)[:, :3]
            import pdb

            # pdb.set_trace()
            initial_q_batch[batch_idx] = initial_q
            robot_pc_batch[batch_idx] = robot_pc
            object_pc_batch[batch_idx] = object_pc

        B, N, DOF = (
            self.batch_size,
            self.num_points,
            len(hand.pk_chain.get_joint_parameter_names()),
        )
        assert initial_q_batch.shape == (B, DOF), (
            f"Expected: {(B, DOF)}, Actual: {initial_q_batch.shape}"
        )
        assert robot_pc_batch.shape == (B, N, 3), (
            f"Expected: {(B, N, 3)}, Actual: {robot_pc_batch.shape}"
        )
        assert object_pc_batch.shape == (B, N, 3), (
            f"Expected: {(B, N, 3)}, Actual: {object_pc_batch.shape}"
        )
        return {
            "robot_name": hand_name,  # str
            "object_name": object_name,  # str
            "initial_q": initial_q_batch,
            "robot_pc": robot_pc_batch,
            "object_pc": object_pc_batch,
        }

    def predict_grasp_pose(
        self, hand_name: str, object_name: str, object_path: str, debug=False
    ):
        data = self.prepare_data(hand_name, object_name, object_path)

        device = self.device
        batch_size = self.batch_size
        cfg = self.cfg

        data_count = 0

        robot_pc = data["robot_pc"].to(device)
        object_pc = data["object_pc"].to(device)
        hand = self.hand_models[hand_name]

        initial_q_list = []
        predict_q_list = []
        object_pc_list = []
        mlat_pc_list = []
        transform_list = []

        while data_count != batch_size:
            split_num = min(batch_size - data_count, cfg.split_batch_size)

            initial_q = data["initial_q"][data_count : data_count + split_num].to(
                device
            )
            robot_pc = data["robot_pc"][data_count : data_count + split_num].to(device)
            object_pc = data["object_pc"][data_count : data_count + split_num].to(
                device
            )

            data_count += split_num

            with torch.no_grad():
                dro = self.network(robot_pc, object_pc)["dro"].detach()

            mlat_pc = multilateration(dro, object_pc)
            transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)
            optim_transform = process_transform(hand.pk_chain, transform)

            layer = create_problem(hand.pk_chain, optim_transform.keys())
            start_time = time.time()
            predict_q = optimization(
                hand.pk_chain,
                layer,
                initial_q,
                optim_transform,  # , n_iter=300
            )
            end_time = time.time()
            print(
                f"[{data_count}/{batch_size}] Optimization time: {end_time - start_time:.4f} s"
            )
            # time_list.append(end_time - start_time)

            initial_q_list.append(initial_q)
            predict_q_list.append(predict_q)
            object_pc_list.append(object_pc)
            mlat_pc_list.append(mlat_pc)
            transform_list.append(transform)

        initial_q_batch = torch.cat(initial_q_list, dim=0)
        predict_q_batch = torch.cat(predict_q_list, dim=0)
        object_pc_batch = torch.cat(object_pc_list, dim=0)
        mlat_pc_batch = torch.cat(mlat_pc_list, dim=0)
        transform_batch = {}
        for transform in transform_list:
            for k, v in transform.items():
                transform_batch[k] = (
                    v
                    if k not in transform_batch
                    else torch.cat((transform_batch[k], v), dim=0)
                )

        # import pdb

        # pdb.set_trace()
        if debug:
            wis3d = Wis3D(
                out_folder="wis3d",
                sequence_name=object_name,
                xyz_pattern=("x", "-y", "-z"),
            )

            object_pc_batch = object_pc_batch.cpu().numpy()
            mlat_pc_batch = mlat_pc_batch.cpu().numpy()
            # robot_pc = robot_pc.cpu().numpy()
            predict_q_batch_clone = predict_q_batch.clone().cpu().numpy()
            initial_q_batch = initial_q_batch.cpu().numpy()

            dro_q_order = hand.get_joint_orders()
            vis_robot = SAPIENKinematicsModelStandalone(hand.urdf_path)
            vis_q_order = [j.name for j in vis_robot.robot.get_active_joints()]

            print("dro_q_order", dro_q_order)
            print("vis_q_order", vis_q_order)
            # import pdb

            # pdb.set_trace()
            dro_to_vis = [dro_q_order.index(name) for name in vis_q_order]

            for i in range(self.batch_size):
                wis3d.set_scene_id(i)
                wis3d.add_point_cloud(object_pc_batch[i], name="object_pc")
                wis3d.add_point_cloud(mlat_pc_batch[i], name="mlat_pc")
                # import pdb

                # pdb.set_trace()
                # wis3d.add_point_cloud(robot_pc[i], name="robot_pc")
                wis3d.add_robot(
                    hand.urdf_path,
                    predict_q_batch_clone[i][dro_to_vis],
                    name="robot_pred",
                )
                # wis3d.add_robot(hand.urdf_path, initial_q_batch[i], name="robot_init")

        return {
            "predict_q": predict_q_batch,
            "object_pc": object_pc_batch,
            "mlat_pc": mlat_pc_batch,
            "predict_transform": transform_batch,
        }

    def get_urdf(self, hand_name: str):
        if hand_name not in self.hand_models:
            self.hand_models[hand_name] = create_hand_model(
                hand_name, device=self.device
            )
        hand = self.hand_models[hand_name]
        return hand.urdf_path

    def get_joint_order(self, hand_name: str):
        if hand_name not in self.hand_models:
            self.hand_models[hand_name] = create_hand_model(
                hand_name, device=self.device
            )
        hand = self.hand_models[hand_name]
        return hand.get_joint_orders()


# This is the main entry point for the script, decorated by Hydra
@hydra.main(version_base="1.2", config_path="configs", config_name="validate")
def main(cfg):  #: DictConfig):
    """
    Main function managed by Hydra.

    The 'cfg' object is automatically populated by Hydra from your config files.
    """
    # Instantiate the main class with the loaded configuration
    proposer = GraspPoseProposal(cfg)

    object_name = [
        "003_cracker_box",
        "006_mustard_bottle",
        "024_bowl",
        "035_power_drill",
        "004_sugar_box",
        "010_potted_meat_can",
    ]
    # Now you can call the methods
    for obj in object_name:
        proposer.predict_grasp_pose(
            hand_name="abilityhand",  # This could also come from cfg
            object_name=obj,
            object_path=f"/home/yulin/Documents/dynamics/Dex-World-Model/dexwm/assets/misc/ycb/visual/{obj}/textured_simple.obj",
        )

    # print("\n--- Final Predicted Poses (Tensor) ---")
    # print(predicted_poses.cpu().numpy())


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)

    main()
