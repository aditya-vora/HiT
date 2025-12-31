import torch
import torch.nn as nn
from einops import rearrange
import numpy as np


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