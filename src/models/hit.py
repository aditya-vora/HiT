import torch
import torch.nn as nn 
from dataclasses import dataclass

from config import Config
from src.models.pointnet import PointNet

@dataclass
class ModelArgs:
    pf_dim: int
    zf_dim: int
    tf_dim: int
    ef_dim: int
    gf_dim: int
    planef_dim: int
    n_planes: int

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

        

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
    


def HiTModelPointNet(**kwargs) -> nn.Module:
    """Factory function to create a HiTModelPointNet instance."""
    model = PT_HiTModel_PointNet(ModelArgs(**kwargs))
    return model

def HiTModelTable(**kwargs) -> nn.Module:
    """Factory function to create a HiTModelTable instance."""
    # Placeholder for actual implementation
    model = nn.Module()  # Replace with actual model
    return model


HiT_models = {
    'hit-pointnet': HiTModelPointNet, 
    'hit-table': HiTModelTable , 
    'hit-volume': HiTModelConvOccNet,
}