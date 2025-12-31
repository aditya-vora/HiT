import torch
import torch.nn as nn 
from dataclasses import dataclass, field
from typing import List

from config import Config
from src.models.pointnet import PointNet
from src.models.conv_occnet import LocalPoolPointnet

@dataclass
class ModelArgs:
    pf_dim: int = 3
    ef_dim: int = 256
    zf_dim: int = 512

    use_xyz: bool = True
    use_bn: bool = False
    plane_type: str = 'grid'
    grid_resolution: int = 32

    tf_dim: int = 4
    gf_dim: int = 256

    n_parts: List[int] = field(default_factory=lambda: [8, 16, 36])
    n_levels: int = 3
    n_planes: int = 32 
    mask_mode: str = "mask"
    planef_dim: int = 8


class PT_HiTModel_PointNet(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super(PT_HiTModel_PointNet, self).__init__()
        """HiT Model Defination
        Args:
            config (Config): Configuration dataclass containing model parameters.
        Returns:
            None
        """

        self.config = config

        self.encoder = PointNet(
            pf_dim=config.pf_dim,
            zf_dim=config.zf_dim,
            ef_dim=config.ef_dim,
            use_xyz=True,
            use_bn=False
        )

        self.decoder = HiTDecoder(config)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
    

class PT_HiTModel_ConvOccNet(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super(PT_HiTModel_ConvOccNet, self).__init__()
        """HiT Model Defination
        Args:
            config (Config): Configuration dataclass containing model parameters.
        Returns:
            None
        """
        unet3d_kwargs = {} 
        unet3d_kwargs['num_levels'] = 3 
        unet3d_kwargs['f_maps'] = 32
        unet3d_kwargs['in_channels'] = config.zf_dim
        unet3d_kwargs['out_channels'] = config.zf_dim

        self.encoder = LocalPoolPointnet(
            zf_dim=config.zf_dim,
            inf_dim=config.pf_dim,
            ef_dim=config.ef_dim,
            use_xyz=config.use_xyz,
            use_bn=config.use_bn,
            plane_type=config.plane_type,
            grid_resolution=config.grid_resolution,
            unet3d_kwargs=unet3d_kwargs,
            unet3d=True
        )

    def forward(self, x):
        # Define forward pass for ConvOccNet here
        # Placeholder for actual implementation
        return x


def HiTModelPointNet(**kwargs) -> nn.Module:
    """function to create a HiTModelPointNet instance."""
    model = PT_HiTModel_PointNet(ModelArgs(**kwargs))
    return model

def HiTModelTable(**kwargs) -> nn.Module:
    """function to create a HiTModelTable instance."""
    # Placeholder for actual implementation
    model = nn.Module()  # Replace with actual model
    return model


def HiTModelConvOccNet(**kwargs) -> nn.Module:
    """function to create a HiTModelConvOccNet instance."""
    model = PT_HiTModel_ConvOccNet(ModelArgs(**kwargs))
    return model


HiT_models = {
    'hit-pointnet': HiTModelPointNet, 
    'hit-table': HiTModelTable , 
    'hit-volume': HiTModelConvOccNet,
}