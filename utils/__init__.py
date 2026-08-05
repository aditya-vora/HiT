import os
from typing import List, Dict, Tuple
from glob import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import json


COLOR_LIST = [
    "255 0 0",     # Fiery Red
    "255 255 0",   # Bright Yellow
    "0 255 0",     # Bright Green
    "0 255 255",   # Cyan
    "0 51 102",    # Deep Blue
    "102 51 153",  # Royal Purple
    "255 51 204",  # Electric Pink
    "255 165 0",   # Orange
    "0 128 128",   # Teal
    "128 0 0",     # Maroon
    "128 128 0",   # Olive
    "0 0 128",     # Navy
    "218 112 214", # Orchid
    "255 127 80",  # Coral
    "106 90 205",  # Slate Blue
    "135 206 235", # Sky Blue
    "255 20 147",  # Deep Pink
    "34 139 34",   # Forest Green
    "70 130 180",  # Steel Blue
    "188 143 143", # Rosy Brown
    "95 158 160",  # Cadet Blue
    "255 99 71",   # Tomato
    "255 140 0",   # Dark Orange
    "147 112 219", # Medium Purple
    "30 144 255",  # Dodger Blue
    "230 230 250", # Lavender
    "47 79 79",    # Dark Slate Gray
    "102 205 170", # Medium Aquamarine
    "184 134 11",  # Dark Goldenrod
    "240 128 128", # Light Coral
    "123 104 238", # Medium Slate Blue
    "0 250 154",   # Medium Spring Green
    "199 21 133",  # Medium Violet Red
    "72 61 139",   # Dark Slate Blue
    "210 105 30",  # Chocolate
    "0 206 209",   # Dark Turquoise
    "138 43 226",  # Blue Violet
]


def break_path(path, delimiter="/") -> List[str]:
    """breaks a path into list of folders

    Args:
        path (str): input path string
        delimiter (str): delimiter to use for breaking the path

    Returns:
        List[str]: list of folders in the path
    """
    folders = path.split(delimiter)
    return folders


def make_path_and_dir(folders: List[str]):
    path = os.path.join(*folders)
    os.makedirs(path, exist_ok=True)
    return path


def get_experiment_index(curr_exp_dir: str) -> int:
    """looks into experiment directory and creates experiment index

    Args:
        curr_exp_dir (str): root directory for experiments

    Returns:
        int: index of the experiment number
    """
    experiment_index = len(glob(f"{curr_exp_dir}/*"))
    return experiment_index


def setup_experiment_directory(base_dir: str, exp_name: str) -> str:
    """takes the base directory path and exp name and create
    folders for logging

    Args:
        base_dir (str): base experiment directory
        exp_name (str): experiment name

    Returns:
        str: experiment directory
    """
    
    exp_dir = make_path_and_dir(folders=[base_dir])
    curr_exp_dir = make_path_and_dir(folders=[exp_dir, exp_name])
    exp_index = get_experiment_index(curr_exp_dir=curr_exp_dir)
    model_string = f"{exp_index:03d}"
    experiment_dir = make_path_and_dir(folders=[curr_exp_dir, model_string])
    return experiment_dir


def read_data_list(
    path: str, 
    names: List[str]=["train", "validation", "test"], 
    return_paths: bool=True
    ) -> List[str]:
    """reads all the lists with prefixs from the names list and location from path.

    Args:
        path (str): string corresponding to location where the text file is stored.
        names (List[str]): List of prefixs of file names to read.
        return_paths (bool): whether to return absolute paths or just file names.
    Returns:
        List[str]: list of absolute paths
    """
    data_list = []
    for name in names:
        if not os.path.exists(os.path.join(path, f"{name}.lst")):
            file_path = os.path.join(path, f"{name}.txt")
            if not os.path.exists(file_path):
                continue
        else:
            file_path = os.path.join(path, f"{name}.lst")
        
        if not os.path.exists(file_path):
            raise ValueError(f"File {file_path} does not exist")
        
        with open(file_path, 'r') as fp:
            lines = [line.strip() for line in fp.readlines()]
        
        for line in lines:
            if return_paths:
                data_list.append(os.path.join(path, line))
            else:
                data_list.append(line)

    return data_list


def numpy_to_pil(images: np.ndarray):
    """
    Convert a NumPy array of shape (batch, height, width, channels) to a list of PIL Images.
    """
    pil_images = []
    for img in images:
        img_uint8 = (img * 255).round().astype("uint8")
        if img_uint8.shape[2] == 1:
            img_uint8 = img_uint8[..., 0]
        pil_images.append(Image.fromarray(img_uint8))
    return pil_images


def map_indexs(indexs: torch.Tensor, pre_indexs: torch.Tensor, g_factor: int) -> torch.Tensor:
    indexs = pre_indexs * g_factor + indexs
    return indexs


def load_checkpoint(model: nn.Module, filename: str) -> None:
    """Loads model weights from a checkpoint saved by train.py (a dict with a "model" key)."""
    checkpoint = torch.load(filename, map_location="cpu")
    model.load_state_dict(checkpoint["model"])


def convert_weights_to_occ(weights: torch.Tensor, occ: torch.Tensor, nparts: int) -> torch.Tensor:
    b, n, _ = occ.shape 
    weights = weights.squeeze(0)
    occ = occ.squeeze(0)
    partocc = torch.zeros(size=(n, nparts), dtype=torch.float32).cuda()
    max_w, max_i = torch.max(weights, dim=-1, keepdim=True)
    partocc.scatter_(1, max_i, occ)
    return partocc.unsqueeze(0)  


def sample_grid(density: int, interval: List[int]) -> torch.Tensor:    
    # create the grid
    x = np.linspace(interval[0], interval[1], density)
    y = np.linspace(interval[0], interval[1], density)
    z = np.linspace(interval[0], interval[1], density)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(
        np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
    return grid


def set_device_and_seed(gpuid: int, seed: int) -> torch.device:
    torch.cuda.set_device(gpuid)
    torch.manual_seed(seed=seed)
    device = torch.device(f"cuda:{gpuid}" if torch.cuda.is_available() else "cpu")
    return device


def read_text(txt_path: str) -> List[str]:
    with open(txt_path, 'r') as fp:
        lines = fp.readlines()
    return [line.strip() for line in lines]


def get_split_shape_ids_from_text_file(path: str) -> List[str]: 
    with open(path, 'r') as f:
        lines = f.readlines()
    shape_ids = [line.strip() for line in lines]
    new_shape_ids = []
    for shape_id in shape_ids: 
        if "." in shape_id:
            shape_id = shape_id.split(".")[0]
        new_shape_ids.append(shape_id)
    return new_shape_ids


def get_split_shape_ids(path: str, split_type: str='train') -> List[str]:
    with open(path, 'r') as f:
        shapes = json.load(fp=f)
    inds = shapes[split_type]
    return inds


def read_json(json_path: str) -> Dict:
    with open(json_path, "r") as file: 
        parts_data = json.load(file)[0]
    return parts_data


def convert_to_torch_tensor(np_arr: np.ndarray, data_type="fp32", device="cuda", batchify: bool=False) -> torch.Tensor:
    if data_type == "fp32":
        return_arry = torch.as_tensor(np_arr, dtype=torch.float32)
    elif data_type == "int64":
        return_arry = torch.as_tensor(np_arr, dtype=torch.int64)
    else:   
        return_arry = torch.from_numpy(np_arr).float()

    if batchify:
        return_arry = return_arry.unsqueeze(0)
    return return_arry.to(device)

    
def convert_to_np_array(tensor: torch.Tensor) -> np.ndarray:
    device = tensor.device
    if device.type == 'cuda':
        return tensor.cpu().numpy()
    else:
        return tensor.numpy()
