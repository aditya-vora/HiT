import torch
import torch.nn as nn 
import torch.nn.functional as F 

from config import DataConfig
from src.dataset.shapenet import ShapeNetDataset

def build_dataset(config: DataConfig, mode="train"):
    # Placeholder for dataset building logic
    if mode == "train":
        if config.dataset_name == "shapenet":
            dataset = ShapeNetDataset(config=config)
            pass
        pass
    pass