import torch 
import torch.nn as nn 
from einops import rearrange, repeat
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from utils.model import convert_euler_angles_to_rotation_matrix

class MultiConvexImplicitDecoder(nn.Module):
    """Multi Convex Implicit Decoder
    Args:
        :param npoints: Number of points in the point cloud
        :param nqpts: Number of query points of the implicit field
        :param zf_dim: Dimension of the embedding
        :param inf_dim: Dimension of the input features
        :param nplanes: Number of planes to be used
        :param planef_dim: Dimension of the planes
    """
    def __init__(
            self, 
            zf_dim: int=4,
            pf_dim: int=3,
            n_planes: int=32,
            planef_dim: int=4, 
            ) -> None:
        super(MultiConvexImplicitDecoder, self).__init__()

        self.zf_dim = zf_dim 
        self.pf_dim = pf_dim

        self.delta_param_dim = 1 
        self.trans_param_dim = pf_dim
        self.exists_dim = 1 
        self.scale_param_dim = 1
        self.rot_param_dim = 3
        self.n_planes = n_planes
        self.planef_dim = planef_dim 

        self.n_convex_global_params = self.delta_param_dim + self.trans_param_dim + self.scale_param_dim + self.rot_param_dim
        self.n_plane_param = pf_dim

        self._offset_scale =  0.5
        self._offset_lbound = 0.0
        self._blend_scale = 250.
        self._blend_lbound = 50.
        self._sharpness = 75.0

        # takes in all the plane features and predicts a global convex parameters
        self.map_to_global_convex_params = nn.Sequential(
            nn.Linear(in_features=zf_dim, out_features=4*zf_dim),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(in_features=4*zf_dim, out_features=self.n_convex_global_params)
        )

        # takes each plane as input and computes its explicit parameters
        self.map_to_plane_params = nn.Sequential(
            nn.Linear(in_features=zf_dim, out_features=4*zf_dim),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(in_features=4*zf_dim, out_features=self.n_plane_param * self.n_planes)
        )

    def _transform_to_local_frame(self, points: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor, scales:torch.Tensor):
        """transform the points to the local frame of the convex
        Args:
            points (torch.Tensor): query points in the 3D space
            rotations (torch.Tensor): euler angles of rotations of the convex. (B, P, 3, 3) 
            translations (torch.Tensor): translation of the convex.
            scales (torch.Tensor): scale of the convex. (B, P, 3, 1)
        :return: transformed points
        """
        DIV_EPSILON = 1e-6
        scales = repeat(scales, 'b p c -> b p (repeat c)', repeat=rotations.shape[-1]) 
        scaled_rots = torch.matmul(torch.diag_embed(1/scales+DIV_EPSILON), rotations)

        points = points[:, None, :, :] - translations[:, :, None, :]
        points = torch.matmul(points, scaled_rots)
        return points


    def _compute_sdf(self, planes: torch.Tensor, translations: torch.Tensor, rotations: torch.Tensor, scales: torch.Tensor, blend_params: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """
        :param planes: planes of the implicit field, [B, Np, Nh, 3]
        :param rotations: rotation euler angle parameters of the convexs , [B, Np, 3]
        :param scales: scale parameters of the convexs, [B, Np, 1] -> [0, inf]
        :param translations: translations of the implicit field, [B, Np, 3]
        :param blend_params: blending parameters of the implicit field, [B, Np, 1]
        :param points: query points of the implicit field, [B, Nq+Np, 3]
        :return: sdf values

        :rtype: torch.Tensor

        :note: each plane is specified by the unit normal (3D) and offset from the origin (1D)

        """
        # compute the sdf values for each point with respect to each plane
        n_parts = planes.shape[1]
        n_planes = planes.shape[2]
        b = planes.shape[0]

        norm_logit = planes[..., :self.inf_dim-1]

        # range [-1, 0]
        offset = -(F.sigmoid(planes[..., self.inf_dim-1:self.inf_dim]) * self._offset_scale + self._offset_lbound)

        blend_params = (F.sigmoid(blend_params[...,:n_parts]) * self._blend_scale + self._blend_lbound) #[50.0, 300.0]

        # norm of the boundary line 
        norm_rad = F.tanh(norm_logit) * np.pi # [..., (azimuth, elevation)]
        norm = torch.stack([
            torch.sin(norm_rad[..., 1]) * torch.cos(norm_rad[..., 0]),
            torch.sin(norm_rad[..., 1]) * torch.sin(norm_rad[..., 0]),
            torch.cos(norm_rad[..., 1])
        ], dim=-1)

        scales = torch.abs(scales)

        # convert the angles to rotation matrices by mapping the angles to range [-pi, pi]
        rotation_matrix = convert_euler_angles_to_rotation_matrix(euler_angles=rotations, map_range=True)        

        points = self._transform_to_local_frame(points, rotations=rotation_matrix, translations=translations, scales=scales)

        # # compute the distance from the point to the plane
        # points = points[:, None, :, :] - translations[:, :, None, :]
        points = torch.tile(points[:,:,None,:,:], [1, 1, n_planes, 1, 1])
        signed_dist = torch.matmul(points, norm[...,None])
        signed_dist = signed_dist + offset[:,:,:,None,:]

        rotation_matrix_flat = rearrange(rotation_matrix, 'b p c d -> b p (c d)')
        transforms = torch.cat([translations, rotation_matrix_flat, scales], dim=-1)
        # transforms = translations
        return signed_dist, transforms, blend_params, norm, offset


    def forward(self, points: torch.Tensor=None, convex_feats: torch.Tensor=None) -> Tuple[torch.Tensor]:
        """
        : params points: query points of the implicit field, [batch_size, n_points, 3]
        : params convex_feats: features of the convex shapes, [batch_size, n_parts, plane_dim * n_planes]
        : return: occupancy values
        """

        B, N, _ = points.shape 
        Np = convex_feats.shape[1]
        nparts = convex_feats.shape[1]
        
        global_convex_params = self.map_to_global_convex_params(convex_feats)
        
        # plane_feats = rearrange(convex_feats, 'b p (n d) -> b p n d', b=B, p=Np, n=self.n_planes, d=self.planef_dim)

        plane_explicit_params = self.map_to_plane_params(convex_feats)
        plane_explicit_params = rearrange(plane_explicit_params, 'b p (n d) -> b p n d', b=B, p=Np, n=self.n_planes, d=self.n_plane_param)
        theta, translations, rotations, scales = torch.split(global_convex_params, split_size_or_sections=[self.delta_param_dim, self.trans_param_dim, self.rot_param_dim, self.scale_param_dim], dim=-1)
        # theta, translations = torch.split(global_convex_params, split_size_or_sections=[self.delta_param_dim, self.trans_param_dim], dim=-1)

        points = torch.concat([points, translations], dim=1)

        signed_dist, transforms, blend_params, normals, offset = self._compute_sdf(
            planes=plane_explicit_params,
            translations=translations,
            rotations=rotations,
            scales=scales,
            blend_params=theta,
            points=points
        )

        # generate convex shapes (use logsumexp as the intersection of halfspaces)
        part_logits = torch.logsumexp(signed_dist * rearrange(blend_params, 'b p c -> b p () () c', b=B, p=nparts), dim=2, keepdim=False)
        part_logits = (-1 * part_logits) / rearrange(blend_params, 'b p c -> b p () c', b=B, p=nparts)

        # generate the occupancy values
        part_indicator_full = F.sigmoid(part_logits * self._sharpness)
        part_indicator = part_indicator_full[:,:,:-nparts]

        # print(torch.min(part_indicator), torch.max(part_indicator))

        shape_indicator_sum = torch.sum(part_indicator_full, dim=1, keepdim=False)
        shape_indicator_max = torch.max(part_indicator, dim=1, keepdim=False)[0]
        
        return shape_indicator_max, (transforms, part_logits[:,:,:-nparts], shape_indicator_sum, normals, offset, part_indicator, blend_params)
