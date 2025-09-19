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
