import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, List
from einops import repeat

from src.models.modules import PointnetSAModule, PointnetFPModule

class PointNet(nn.Module):
    """Some Information about PointNetCapsule"""
    def __init__(
            self, 
            pf_dim: int=3, 
            zf_dim: int=32, 
            ef_dim:int=64, 
            use_xyz: bool=True, 
            use_bn: bool=False, 
            abst_pts_list: List[int]=[1024,256,64,16],
            abst_rad_list: List[float]=[0.1,0.2,0.4,0.8],
            knn: int=32
        ) -> None:
        super(PointNet, self).__init__()
        """PointNet Model Defination
        Args:
            pf_dim (int): Point feature dimension.
            zf_dim (int): Latent feature dimension.
            ef_dim (int): Encoder hidden dimension.
            use_xyz (bool): Whether to use XYZ coordinates as features.
            use_bn (bool): Whether to use batch normalization.
            abst_pts_list (List[int]): List of abstracted points at each SA layer.
            abst_rad_list (List[float]): List of radii for each SA layer.
            knn (int): Number of nearest neighbors to consider.
        Returns:
            None
        """

        self.SA_modules, self.FP_modules = nn.ModuleList([]), nn.ModuleList([])
        self.SA_modules.append(PointnetSAModule(
            npoint=abst_pts_list[0], 
            radius=abst_rad_list[0], 
            nsample=knn, 
            mlp=[pf_dim, ef_dim, ef_dim, ef_dim*2], 
            use_xyz=use_xyz, 
            bn=use_bn))
        
        self.SA_modules.append(PointnetSAModule(
            npoint=abst_pts_list[1], 
            radius=abst_rad_list[1], 
            nsample=knn, 
            mlp=[ef_dim*2, ef_dim*2, ef_dim*2, ef_dim*4], 
            use_xyz=use_xyz, 
            bn=use_bn))
        
        self.SA_modules.append(PointnetSAModule(
            npoint=abst_pts_list[2], 
            radius=abst_rad_list[2], 
            nsample=knn, 
            mlp=[ef_dim*4, ef_dim*4, ef_dim*4, ef_dim*8], 
            use_xyz=use_xyz, 
            bn=use_bn))
        
        self.SA_modules.append(PointnetSAModule(
            npoint=abst_pts_list[3], 
            radius=abst_rad_list[3], 
            nsample=knn, 
            mlp=[ef_dim*8, ef_dim*8, ef_dim*8, ef_dim*16], 
            use_xyz=use_xyz, 
            bn=use_bn))

        self.FP_modules.append(PointnetFPModule(
            mlp=[ef_dim*4 + pf_dim, ef_dim*4, ef_dim*4, ef_dim*4], 
            bn=use_bn))
        
        self.FP_modules.append(PointnetFPModule(
            mlp=[ef_dim*8 + ef_dim*2, ef_dim*8, ef_dim*4], 
            bn=use_bn))
        
        self.FP_modules.append(PointnetFPModule(
            mlp=[ef_dim*8 + ef_dim*4, ef_dim*8, ef_dim*8], 
            bn=use_bn))
        
        self.FP_modules.append(PointnetFPModule(
            mlp=[ef_dim*16 + ef_dim*8, ef_dim*8, ef_dim*8], 
            bn=use_bn))
    
        self.fc_layer = nn.Sequential(
            nn.Conv1d(ef_dim*4, ef_dim*4, 1, bias=False),
            nn.BatchNorm1d(ef_dim*4),
            nn.ReLU(),
            nn.Conv1d(ef_dim*4, ef_dim*8, 1, bias=False),
            nn.BatchNorm1d(ef_dim*8),
            nn.Conv1d(ef_dim*8, zf_dim, 1, bias=False),
            nn.BatchNorm1d(zf_dim),
        )

    def _break_up_pc(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return (xyz, features)

    def forward(self, pointcloud: torch.Tensor) -> torch.Tensor:

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        per_point_feats = self.fc_layer(l_features[0])
        per_point_feats = per_point_feats.permute(0,2,1)

        return per_point_feats