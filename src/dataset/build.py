import torch
import torch.nn as nn 
import torch.nn.functional as F 

from config import DataConfig
from torch.utils.data import Dataset

from src.dataset.shapenet import ShapeNetDataset

def build_dataset(
        config: DataConfig, 
        train_mode: bool=False, 
        recon_mode: bool=False, 
        pc_mode: bool=False, 
        mesh_mode: bool=False,
        iou_mode: bool=False,
        cd_mode: bool=False
    ) -> Dataset:
    # Placeholder for dataset building logic
    if train_mode:
        if config.dataset_name == "shapenet":
            return ShapeNetDataset(config=config, train_mode=train_mode, recon_mode=recon_mode, pc_mode=pc_mode, mesh_mode=mesh_mode, iou_mode=iou_mode, cd_mode=cd_mode)

    raise ValueError(f'dataset {config.dataset_name} is not supported')