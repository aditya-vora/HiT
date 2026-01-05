import os 
import trimesh 
from typing import Dict
import numpy as np

from utils import read_json


def parse_parts_info(parts_info: Dict=None, map_dict: Dict=None):
    """parses the parts info from the json file
    Args:
        parts_info (Dict, optional): Dictionary containing parts information. Defaults to None.
        map_dict (Dict, optional): Dictionary to store parsed parts information. Defaults to None.
    Returns:
        List: List of object names under the current part
    """
    partid = parts_info['id']
    map_dict[partid] = {}
    map_dict[partid]['name'] = parts_info['name']
    if 'children' in parts_info:
        objs = []
        for child in parts_info['children']:
            objs += parse_parts_info(child, map_dict)
    else:
        objs = [s for s in parts_info['objs']]
    map_dict[partid]['objs'] = objs
    return objs


def read_full_mesh_from_part_json(json_path: str, obj_root_path: str) -> trimesh.Trimesh:
    """reads the full mesh from a part json file
    Args:
        json_path (str): path to the part json file
        obj_root_path (str): root path to the obj files
    Returns:
        trimesh.Trimesh: full mesh in trimesh format
    """ 
    part_json_info = read_json(json_path=json_path)
    part_dict = dict() 
    parse_parts_info(parts_info=part_json_info, map_dict=part_dict)
    obj_names = part_dict[0]['objs']
    part_trimesh_objs = [] 
    for obj_name in obj_names:
        obj_filepath = os.path.join(obj_root_path, f"{obj_name}.obj")
        obj = trimesh.load(obj_filepath, force="mesh", process=False)
        part_trimesh_objs.append(obj)
    return trimesh.util.concatenate(part_trimesh_objs)


def scale_mesh_unit_sphere(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """scales the mesh to fit inside a unit sphere
    Args:
        mesh (trimesh.Trimesh): input mesh in trimesh format
    Returns:
        trimesh.Trimesh: scaled mesh in trimesh format
    """
    vertices = mesh.vertices    
    x_max, y_max, z_max = np.max(vertices, axis=0)
    x_min, y_min, z_min = np.min(vertices, axis=0)
    x_center, y_center, z_center = (x_max + x_min) / 2, (y_max + y_min) / 2, (z_max + z_min) / 2
    x_extent, y_extent, z_extent = x_max - x_min, y_max - y_min, z_max - z_min
    scale = np.sqrt(x_extent ** 2 + y_extent ** 2 + z_extent ** 2)
    vertices -= np.array([x_center, y_center, z_center])
    vertices /= scale
    mesh.vertices = vertices
    return mesh 

