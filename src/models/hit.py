import torch
import torch.nn as nn 

from config import Config

class HiTModel(nn.Module):
    def __init__(self, config):
        super(HiTModel, self).__init__()
        self.config = config
        # Example layers based on config parameters
        self.input_layer = nn.Linear(config.pf_dim, config.zf_dim)
        self.hidden_layer = nn.Linear(config.zf_dim, config.ef_dim)
        self.output_layer = nn.Linear(config.ef_dim, config.gf_dim)
        # More layers and components can be added here based on the architecture

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.relu(x)
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x