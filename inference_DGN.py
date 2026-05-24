import os
import sys
import glob
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

from utils.wis3d_new import Wis3D, SAPIENKinematicsModelStandalone
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
        self,
        hand_name: str,
        object_name: str,
        object_path: str,
        object_scale=None,
        object_pose=None,
        initial_q=None,
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
                # cache key includes scale and (if provided) pose so the same
                # mesh at different scales/poses writes to different cache files
                tag_parts = []
                if object_scale is not None:
                    scale_arr = np.asarray(object_scale, dtype=np.float32).reshape(-1)
                    tag_parts.append(
                        "scale" + "_".join(f"{s:.4f}" for s in scale_arr)
                    )
                if object_pose is not None:
                    pose_arr = np.asarray(object_pose, dtype=np.float32).reshape(-1)
                    tag_parts.append(
                        "pose" + "_".join(f"{p:.4f}" for p in pose_arr)
                    )
                cache_tag = ("_" + "_".join(tag_parts)) if tag_parts else ""
                object_pcld_path = object_path.replace(
                    ".obj", f"{cache_tag}_pcd.ply"
                )
                if not os.path.exists(object_pcld_path):
                    mesh = trimesh.load_mesh(object_path)
                    if object_scale is not None:
                        scale_arr = np.asarray(object_scale, dtype=np.float32).reshape(
                            -1
                        )
                        if scale_arr.size == 1:
                            mesh.apply_scale(float(scale_arr[0]))
                        else:
                            T = np.eye(4)
                            T[0, 0] = float(scale_arr[0])
                            T[1, 1] = float(scale_arr[1])
                            T[2, 2] = float(scale_arr[2])
                            mesh.apply_transform(T)
                    if object_pose is not None:
                        # pose convention: [tx, ty, tz, qw, qx, qy, qz].
                        # We deliberately apply only the rotation so the object
                        # PC stays near zero-mean (which is what the network
                        # was trained on). Callers using a seed must subtract
                        # the scene-cfg translation themselves.
                        from scipy.spatial.transform import Rotation as _R
                        pose_arr = np.asarray(object_pose, dtype=np.float64).reshape(-1)
                        assert pose_arr.size == 7, (
                            f"object_pose must be length 7 [tx,ty,tz,qw,qx,qy,qz], got {pose_arr.size}"
                        )
                        qw, qx, qy, qz = pose_arr[3:7]
                        rot = _R.from_quat([qx, qy, qz, qw]).as_matrix()
                        T = np.eye(4)
                        T[:3, :3] = rot
                        mesh.apply_transform(T)
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

        # If a fixed initial_q is supplied, use it instead of random sampling.
        # Accepted shapes (euler representation, same as hand.get_initial_q()):
        #   (hand.dof,)            -> broadcast to every batch element
        #   (self.batch_size, dof) -> one seed per batch element
        fixed_initial_q = None
        if initial_q is not None:
            if isinstance(initial_q, np.ndarray):
                fixed_initial_q = torch.tensor(initial_q, dtype=torch.float32)
            elif torch.is_tensor(initial_q):
                fixed_initial_q = initial_q.detach().to(torch.float32).cpu()
            else:
                fixed_initial_q = torch.tensor(initial_q, dtype=torch.float32)
            if fixed_initial_q.ndim == 1:
                assert fixed_initial_q.shape == (hand.dof,), (
                    f"initial_q must have shape ({hand.dof},), got {tuple(fixed_initial_q.shape)}"
                )
            elif fixed_initial_q.ndim == 2:
                assert fixed_initial_q.shape == (self.batch_size, hand.dof), (
                    f"initial_q must have shape ({self.batch_size}, {hand.dof}), "
                    f"got {tuple(fixed_initial_q.shape)}"
                )
            else:
                raise ValueError(
                    f"initial_q must be 1-D or 2-D, got {fixed_initial_q.ndim}-D"
                )

        for batch_idx in range(self.batch_size):
            if fixed_initial_q is not None:
                if fixed_initial_q.ndim == 1:
                    initial_q_b = fixed_initial_q.clone()
                else:
                    initial_q_b = fixed_initial_q[batch_idx].clone()
            else:
                initial_q_b = hand.get_initial_q()
            robot_pc = hand.get_transformed_links_pc(initial_q_b)[:, :3]

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
            initial_q_batch[batch_idx] = initial_q_b
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
        self,
        hand_name: str,
        object_name: str,
        object_path: str,
        object_scale=None,
        object_pose=None,
        initial_q=None,
        debug=False,
    ):
        data = self.prepare_data(
            hand_name,
            object_name,
            object_path,
            object_scale=object_scale,
            object_pose=object_pose,
            initial_q=initial_q,
        )

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
        # debug=True
        if debug:
            safe_object_name = object_name.replace("/", "_")
            wis3d = Wis3D(
                out_folder="wis3d",
                sequence_name=f"dro_{safe_object_name}",
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

            for i in range(max(self.batch_size, 4)):
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
                wis3d.add_robot(
                    hand.urdf_path,
                    initial_q_batch[i][dro_to_vis],
                    name="robot_init",
                )

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


DGN_ROOT = os.path.join(ROOT_DIR, "data/object/DGN_2k_origin")
DGN_SCENE_CFG_ROOT = os.path.join(DGN_ROOT, "scene_cfg")
DGN_PROCESSED_ROOT = os.path.join(DGN_ROOT, "processed_data")


def load_dgn_scene_cfg(npy_path: str):
    """Load a DGN scene_cfg .npy and return (object_name, mesh_path, scale, info).

    The .npy stores a dict with `scene[obj_name]` containing a relative
    `file_path` (mesh) and per-axis `scale`. Mesh path is resolved against the
    processed_data tree so we can pass an absolute .obj path to prepare_data.
    """
    data = np.load(npy_path, allow_pickle=True).item()
    scene_id = data["scene_id"]
    # obj_name comes from the task entry; fall back to the first non-table scene key
    obj_name = data.get("task", {}).get("obj_name")
    if obj_name is None:
        for k in data["scene"].keys():
            if k != "table":
                obj_name = k
                break
    obj_entry = data["scene"][obj_name]

    # Mesh path is stored relative to scene_cfg/<obj>/<robot>/, but we just want
    # the obj's processed mesh. Use simplified.obj if it exists, else fall back
    # to normalized.obj.
    candidate_simplified = os.path.join(
        DGN_PROCESSED_ROOT, obj_name, "mesh", "simplified.obj"
    )
    candidate_normalized = os.path.join(
        DGN_PROCESSED_ROOT, obj_name, "mesh", "normalized.obj"
    )
    if os.path.exists(candidate_simplified):
        mesh_path = candidate_simplified
    else:
        mesh_path = candidate_normalized

    scale = np.asarray(obj_entry["scale"], dtype=np.float32)
    # import pdb; pdb.set_trace()
    pose = np.asarray(obj_entry.get("pose", np.zeros(7)), dtype=np.float32)
    return {
        "scene_id": scene_id,
        "object_name": obj_name,
        "mesh_path": mesh_path,
        "scale": scale,
        "pose": pose,
        "raw": data,
    }


# This is the main entry point for the script, decorated by Hydra
@hydra.main(version_base="1.2", config_path="configs", config_name="validate")
def main(cfg):  #: DictConfig):
    """
    Main function managed by Hydra.

    Loads each scene_cfg .npy under DGN_2k_origin and predicts grasps for the
    referenced object mesh at the recorded scale.
    """
    proposer = GraspPoseProposal(cfg)

    # Allow overriding which scene_cfg files to run via cfg if provided,
    # otherwise default to the single example requested.
    npy_paths = getattr(cfg, "dgn_npy_paths", None)
    if npy_paths is None:
        default_path = os.path.join(
            DGN_SCENE_CFG_ROOT,
            "core_bottle_1a7ba1f4c892e2da30711cdbdbc73924",
            "tabletop_ur10e",
            "scale006_pose000_0.npy",
        )
        npy_paths = [default_path]
    elif isinstance(npy_paths, str):
        # treat as a glob pattern
        npy_paths = sorted(glob.glob(npy_paths))

    hand_name = getattr(cfg, "hand_name", "fixsharpa_right")
    debug = bool(getattr(cfg, "debug", False))

    results = {}
    for npy_path in npy_paths:
        info = load_dgn_scene_cfg(npy_path)
        if not os.path.exists(info["mesh_path"]):
            print(f"[skip] mesh not found: {info['mesh_path']}")
            continue
        # Use scene_id as the cache key so the same object at different scales
        # does not collide in self.object_pcs.
        cache_key = info["scene_id"]
        print(
            f"[run] {cache_key}  mesh={info['mesh_path']}  scale={info['scale']}"
        )
        out = proposer.predict_grasp_pose(
            hand_name=hand_name,
            object_name=cache_key,
            object_path=info["mesh_path"],
            object_scale=info["scale"],
            object_pose=info["pose"],
            debug=debug,
        )
        results[cache_key] = out

    return results


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=FutureWarning)
    torch.set_num_threads(8)

    main()

"""
# single object, all poses
  python inference_DGN.py '+dgn_npy_paths="data/object/DGN_2k_origin/scene_cfg/core_bottle_1a7ba1f4
  c892e2da30711cdbdbc73924/tabletop_ur10e/*.npy"'

  # every bottle, every pose
  python inference_DGN.py '+dgn_npy_paths="data/object/DGN_2k_origin/scene_cfg/*/tabletop_ur10e/*.npy"'

  # enable wis3d debug
  python inference_DGN.py '+debug=true'
"""