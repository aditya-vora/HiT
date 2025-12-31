import torch
import torch.nn as nn
from einops import rearrange
import numpy as np


# UNet3D utils.
def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_coordinate(p, padding=0.1, plane='xz'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        plane (str): plane feature type, ['xz', 'xy', 'yz']
    '''
    if plane == 'xz':
        xy = p[:, :, [0, 2]]
    elif plane =='xy':
        xy = p[:, :, [0, 1]]
    else:
        xy = p[:, :, [1, 2]]

    xy_new = xy / (1 + padding + 10e-6) # (-0.5, 0.5)
    xy_new = xy_new + 0.5 # range (0, 1)

    # f there are outliers out of the range
    if xy_new.max() >= 1:
        xy_new[xy_new >= 1] = 1 - 10e-6
    if xy_new.min() < 0:
        xy_new[xy_new < 0] = 0.0
    return xy_new


def normalize_3d_coordinate(p, padding=0.0):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    # print(torch.min(p), torch.max(p))
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    # p_nor = p_nor + 0.5 # range (0, 1)
    p_nor = (p_nor + 1.0) / 2.0

    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def convert_euler_angles_to_rotation_matrix(euler_angles: torch.Tensor, map_range: bool=True) -> torch.Tensor:    
    """
    Convert Euler angles to rotation matrix.
    :param euler_angles: Tensor of shape (B, N, 3) [-pi, pi]. containing Euler angles in radians.
    :return: Tensor of shape (B, N, 3, 3) containing rotation matrices.
    """

    if map_range:
        euler_angles = (torch.sigmoid(euler_angles) - 0.5) * 2 * np.pi
    
    B, Np = euler_angles.shape[:2]
    cosines, sines = torch.cos(euler_angles), torch.sin(euler_angles)

    cx, cy, cz = cosines.unbind(dim=-1)
    sx, sy, sz = sines.unbind(dim=-1)

    # rotation matrix
    rotations = torch.stack(
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
        sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
        -sy, cy * sx, cy * cx], dim=-1)

    rotations = rearrange(rotations, 'b p (i j) -> b p i j', b=B, p=Np, i=3, j=3)
    return rotations 