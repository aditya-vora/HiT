import os 
import trimesh 
from typing import Dict
import numpy as np
import torch

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


def face_areas_normals(faces, vs):
    face_normals = torch.cross(
        vs[:, faces[:, 1], :] - vs[:, faces[:, 0], :],
        vs[:, faces[:, 2], :] - vs[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2) + 1e-8
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


def sample_surface_torch(faces, vs, count):
    
    if torch.isnan(faces).any() or torch.isnan(vs).any():
        assert False, 'saw nan in sample_surface'

    device = vs.device
    bsize, nvs, _ = vs.shape
    area, normal = face_areas_normals(faces, vs)
    area_sum = torch.sum(area, dim=1)

    assert not (area <= 0.0).any().item(
    ), "Saw negative probability while sampling"
    assert not (area_sum <= 0.0).any().item(
    ), "Saw negative probability while sampling"
    assert not (area > 1000000.0).any().item(), "Saw inf"
    assert not (area_sum > 1000000.0).any().item(), "Saw inf"

    dist = torch.distributions.categorical.Categorical(
        probs=area / (area_sum[:, None]))
    
    face_index = dist.sample((count,))
    keep_face_index = face_index.clone()
    
    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vs[:, faces[:, 0], :]
    tri_vectors = vs[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(
        1,
        1,
        2
    ).reshape((bsize, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((bsize, count, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((bsize, count, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(
        count, 2, 1, device=vs.device, dtype=tri_vectors.dtype)

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    normals = torch.gather(normal, dim=1, index=face_index)[0]
    
    return samples[0], keep_face_index.squeeze(), normals