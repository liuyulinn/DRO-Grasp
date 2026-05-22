from wis3d import Wis3D
import trimesh
import sapien
import numpy as np
import warnings
import transforms3d

import torch
import numpy as np
from multipledispatch import dispatch


def Rt_to_pose(R, t=np.zeros(3)):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def cart_to_hom(pts):
    """
    :param pts: (N, 3 or 2)
    :return pts_hom: (N, 4 or 3)
    """
    if isinstance(pts, np.ndarray):
        pts_hom = np.concatenate(
            (pts, np.ones([*pts.shape[:-1], 1], dtype=np.float32)), -1
        )
    else:
        ones = torch.ones([*pts.shape[:-1], 1], dtype=torch.float32, device=pts.device)
        pts_hom = torch.cat((pts, ones), dim=-1)
    return pts_hom


def hom_to_cart(pts):
    return pts[..., :-1] / pts[..., -1:]


@dispatch(np.ndarray, np.ndarray)
def transform_points(pts, pose):
    """
    :param pts: Nx3
    :param pose: 4x4
    :param calib:Calibration
    :return:
    """
    pts = cart_to_hom(pts)
    pts = pts @ pose.T
    pts = hom_to_cart(pts)
    return pts


@dispatch(torch.Tensor, torch.Tensor)
def transform_points(pts, pose):
    pts = cart_to_hom(pts)
    pts = pts @ pose.transpose(-1, -2)
    pts = hom_to_cart(pts)
    return pts


def depth_to_cam(K, depth_map):
    """

    :param K: np.ndarray or torch.Tensor, 3x3
    :param depth_map: H,W
    :return:
    """
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(depth_map, np.ndarray):
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
    else:
        x_range = torch.arange(0, depth_map.shape[1]).to(device=depth_map.device)
        y_range = torch.arange(0, depth_map.shape[0]).to(device=depth_map.device)
        y_idxs, x_idxs = torch.meshgrid(y_range, x_range, indexing="ij")
    x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
    depth = depth_map[y_idxs, x_idxs]
    pts_rect = img_to_cam(K, x_idxs + 0.5, y_idxs + 0.5, depth)
    return pts_rect


def img_to_cam(K, u, v, depth_rect):
    """
    :param u: (N)
    :param v: (N)
    :param depth_rect: (N)
    :return: pts_rect:(N, 3)
    """
    # check_type(u)
    # check_type(v)

    fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if isinstance(depth_rect, np.ndarray):
        x = ((u - cu) * depth_rect) / fu
        y = ((v - cv) * depth_rect) / fv
        pts_rect = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1
        )
    else:
        x = ((u.float() - cu) * depth_rect) / fu
        y = ((v.float() - cv) * depth_rect) / fv
        pts_rect = torch.cat(
            (x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), dim=1
        )
    return pts_rect


class SAPIENKinematicsModelStandalone:
    def __init__(self, urdf_path, srdf_path=None):
        self.engine = sapien.Engine()

        self.scene = self.engine.create_scene()

        loader = self.scene.create_urdf_loader()
        # loader.scale = 10
        builder = loader.load_file_as_articulation_builder(urdf_path, srdf_path)
        # builder = loader.load(urdf_path)

        self.robot: sapien.Articulation = builder.build(fix_root_link=True)

        self.robot.set_pose(sapien.Pose())

        self.robot.set_qpos(np.zeros(self.robot.dof))

        self.scene.step()

        self.model: sapien.PinocchioModel = self.robot.create_pinocchio_model()

    def compute_forward_kinematics(self, qpos, link_index) -> sapien.Pose:
        # print('left 6!!!!')
        if len(qpos) != self.robot.dof:
            warnings.warn("qpos length not match")
        qpos = np.array(qpos).tolist() + [0] * (self.robot.dof - len(qpos))
        # print('in fk', qpos, link_index)
        # self.robot.set_qpos(qpos)
        self.model.compute_forward_kinematics(np.array(qpos))

        return self.model.get_link_pose(link_index)
        # for i in self.robot.links:
        #     if i.name == 'Link_left6':
        #         return i.pose

    #
    def compute_inverse_kinematics(
        self, link_index, pose, initial_qpos, *args, **kwargs
    ):
        if len(initial_qpos) != self.robot.dof:
            warnings.warn("initial_qpos length not match")
        initial_qpos = np.array(initial_qpos).tolist() + [0] * (
            self.robot.dof - len(initial_qpos)
        )

        return self.model.compute_inverse_kinematics(
            link_index, pose, initial_qpos=initial_qpos, *args, **kwargs
        )

    def release(self):
        self.scene = None

        self.engine = None


class Wis3D(Wis3D):
    urdf_caches = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_robot(
        self,
        urdf_path,
        qpos=None,
        Tw_w2B=None,
        add_local_coord=False,
        name="",
        return_joint_mesh=False,
    ):
        """
        add robot.
        Parameters
        ----------
        urdf_path: str. path to urdf file
        qpos: len(active_joints)-dim ndarray. radian. default to be 0. If len(qpos) < len(active_joints), the rest will be filled with 0.
        Tw_w2B: world to base transformation.
        add_local_coord: whether to visualize per-link local coordinate system.
        name: keep this empty unless you are visualizing multiple robots.

        Returns
        -------

        """
        if not self.enable:
            return
        try:
            import sapien
        except ImportError:
            raise ImportError("Please install sapien first.")
        if urdf_path not in Wis3D.urdf_caches:
            Wis3D.urdf_caches[urdf_path] = SAPIENKinematicsModelStandalone(urdf_path)
        if Tw_w2B is None:
            Tw_w2B = np.eye(4)

        sk: SAPIENKinematicsModelStandalone = Wis3D.urdf_caches[urdf_path]
        links = sk.robot.get_links()
        if qpos is None:
            warnings.warn("qpos is not provided, using default qpos.")
            qpos = np.zeros(len(sk.robot.get_active_joints()))
        if len(qpos) < len(sk.robot.get_active_joints()):
            warnings.warn(
                f"qpos is not complete {len(qpos)} < {len(sk.robot.get_active_joints())}, filling the rest with 0."
            )
            qpos = np.asarray(qpos).tolist() + [0] * (
                len(sk.robot.get_active_joints()) - len(qpos)
            )
        local_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.05
        joint_mesh = None
        for link_index, link in enumerate(links):
            link_name = link.name
            pq = sk.compute_forward_kinematics(qpos, link_index)

            pose = Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
            components = link.get_entity().get_components()
            for component in components:
                if isinstance(component, sapien.pysapien.render.RenderBodyComponent):
                    local_pose = component.render_shapes[
                        0
                    ].local_pose.to_transformation_matrix()
                    pose = pose @ local_pose
                    if hasattr(component.render_shapes[0], "filename"):
                        mesh_path = component.render_shapes[0].filename
                        mesh = trimesh.load(mesh_path, force="mesh")
                        add_name = link_name if name == "" else name + "_" + link_name
                        m = trimesh.Trimesh(
                            transform_points(mesh.vertices, Tw_w2B @ pose),
                            mesh.faces,
                        )
                        if return_joint_mesh:
                            if joint_mesh is None:
                                joint_mesh = m
                            else:
                                joint_mesh = trimesh.util.concatenate(joint_mesh, m)
                        self.add_mesh(m, name=add_name)
                        axis_in_base = transform_points(
                            local_pts, Tw_w2B @ pose
                        )
                        if add_local_coord:
                            self.add_coordinate_transformation(
                                Tw_w2B @ pose, name=f"{link_name}_coord"
                            )
                            # self.add_lines(axis_in_base[0], axis_in_base[1], name=f'{link_name}_x')
                            # self.add_lines(axis_in_base[0], axis_in_base[2], name=f'{link_name}_y')
                            # self.add_lines(axis_in_base[0], axis_in_base[3], name=f'{link_name}_z')
        if return_joint_mesh:
            return joint_mesh

    def add_coordinate_transformation(self, Tw_w2B, name="coordinate", scale=0.05):
        ## visualize the coordinate transformation
        ## visualize 3 axes, using self.add_lines
        # start_points = Tw_w2B[:3, 3:4]
        origin = Tw_w2B[:3, 3]  # Translation component (origin in world frame)

        # Define unit axes in local frame, scaled
        x_axis = Tw_w2B[:3, :3] @ (np.array([1, 0, 0]) * scale) + origin
        y_axis = Tw_w2B[:3, :3] @ (np.array([0, 1, 0]) * scale) + origin
        z_axis = Tw_w2B[:3, :3] @ (np.array([0, 0, 1]) * scale) + origin

        # Add the lines representing the axes
        self.add_lines(
            origin, x_axis, colors=[[255, 0, 0]], name=f"{name}_x"
        )  # X axis (red)
        self.add_lines(
            origin, y_axis, colors=[[0, 255, 0]], name=f"{name}_y"
        )  # Y axis (green)
        self.add_lines(
            origin, z_axis, colors=[[0, 0, 255]], name=f"{name}_z"
        )  # Z axis (blue)

    ## -----------------------------------------------------------------##
    ## --                NEW SPHERE VISUALIZATION CODE                  --##
    ## -----------------------------------------------------------------##
    def add_collision_spheres(
        self,
        urdf_path,
        collision_data,
        qpos=None,
        Tw_w2B=None,
        name_prefix="",
        color=[0, 255, 0, 150]
    ):
        """
        Visualizes collision spheres on the robot.

        Parameters
        ----------
        urdf_path: str. Path to the robot's URDF file.
        collision_data: dict. Parsed from YAML, mapping link names to sphere data.
        qpos: np.ndarray. Joint positions of the robot.
        Tw_w2B: np.ndarray. 4x4 transformation from base to world.
        name_prefix: str. Prefix for the visualized sphere names.
        color: list. RGBA color for the spheres.
        """
        # from utils import utils_3d
        
        if not self.enable:
            return

        # Ensure the robot kinematics model is loaded and cached.
        if urdf_path not in Wis3D.urdf_caches:
            Wis3D.urdf_caches[urdf_path] = SAPIENKinematicsModelStandalone(urdf_path)

            # print(f"Kinematics model for {urdf_path} not cached. Loading it now.")
            # self.add_robot(urdf_path, qpos, Tw_w2B, name=name_prefix)

        sk: SAPIENKinematicsModelStandalone = Wis3D.urdf_caches[urdf_path]
        link_names_from_model = [link.get_name() for link in sk.robot.get_links()]

        if qpos is None:
            qpos = np.zeros(len(sk.robot.get_active_joints()))
        if len(qpos) < len(sk.robot.get_active_joints()):
            qpos = np.asarray(qpos).tolist() + [0] * (len(sk.robot.get_active_joints()) - len(qpos))

        if Tw_w2B is None:
            Tw_w2B = np.eye(4)

        print("Visualizing collision spheres...")
        for link_name, spheres in collision_data.items():
            if link_name not in link_names_from_model:
                warnings.warn(f"Link '{link_name}' from YAML not in URDF model. Skipping.")
                continue

            link_index = link_names_from_model.index(link_name)

            # Get the world pose of the link frame
            pq = sk.compute_forward_kinematics(qpos, link_index)
            link_pose = Rt_to_pose(transforms3d.quaternions.quat2mat(pq.q), pq.p)
            world_link_pose = Tw_w2B @ link_pose

            for i, sphere_data in enumerate(spheres):
                center_local = np.array(sphere_data["center"])
                radius = sphere_data["radius"]

                # Create a trimesh sphere at the origin
                sphere_mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
                
                # Transform the sphere's local center to the world frame
                world_center = transform_points(center_local, world_link_pose).flatten()
                
                # Move the sphere mesh to its final position
                sphere_mesh.apply_translation(world_center)

                # Assign color to the mesh.
                sphere_mesh.visual.vertex_colors = color

                # Add the transformed mesh to the visualizer
                full_name = f"{name_prefix}{link_name}_sphere_{i}"
                self.add_mesh(sphere_mesh, name=full_name)
        print("Done visualizing collision spheres.")