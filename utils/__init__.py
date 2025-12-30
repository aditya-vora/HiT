import os 
from typing import List
from glob import glob
import numpy as np
from PIL import Image

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