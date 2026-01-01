import torch
import torch.nn as nn 
import torch.nn.functional as F 

from config import Config, LossConfig
from src.dataset.shapenet import ShapeNetDatasetLazy

def build_dataset(config: Config, mode="train"):
    # Placeholder for dataset building logic
    if mode == "train":
        if config.dataset_name == "shapenet":
            dataset = ShapeNetDatasetLazy(config=config)
            pass
        pass
    pass