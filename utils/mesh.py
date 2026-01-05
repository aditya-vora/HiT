import os 
import trimesh 
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance
from typing import Tuple, List
from scipy.ndimage import binary_erosion, binary_dilation

from utils import read_json, convert_to_torch_tensor


def extract_surface_voxels(voxels: np.array=None) -> np.array:
    voxels_eroded = binary_erosion(voxels)
    voxels_dilated = binary_dilation(voxels)
    surface_voxels = (voxels_dilated ^ voxels_eroded)
    return surface_voxels


def sample_in_band(dim: int, band_min: float, band_max: float, nsamples: int, s_range=[-0.6,0.6]) -> torch.Tensor:
    samples = torch.rand((nsamples, 3)) * (s_range[1] - s_range[0]) + s_range[0]
    samples[:, dim] = torch.rand((nsamples,)) * (band_max - band_min) + band_min
    return samples


def pad_volume(points: torch.Tensor, occ: torch.Tensor, nsample: int, sample_range: List[float]=[-0.5,0.5], pad=0.1) -> None:
    new_sample_min, new_sample_max = sample_range[0] - pad, sample_range[1] + pad

    band_min_1, band_max_1 = sample_range[0] - pad, sample_range[0]
    band_min_2, band_max_2 = sample_range[1], sample_range[1] + pad

    nsamples_per_band = nsample // 6

    # Sample points for each of the six bands
    new_query_coords = torch.cat([
        sample_in_band(0, band_min_1, band_max_1, nsamples_per_band, s_range=[new_sample_min, new_sample_max]),  # X-axis negative band
        sample_in_band(0, band_min_2, band_max_2, nsamples_per_band, s_range=[new_sample_min, new_sample_max]),  # X-axis positive band
        sample_in_band(1, band_min_1, band_max_1, nsamples_per_band, s_range=[new_sample_min, new_sample_max]),  # Y-axis negative band
        sample_in_band(1, band_min_2, band_max_2, nsamples_per_band, s_range=[new_sample_min, new_sample_max]),  # Y-axis positive band
        sample_in_band(2, band_min_1, band_max_1, nsamples_per_band, s_range=[new_sample_min, new_sample_max]),  # Z-axis negative band
        sample_in_band(2, band_min_2, band_max_2, nsamples_per_band, s_range=[new_sample_min, new_sample_max])   # Z-axis positive band
    ], dim=0)

    new_query_coords = new_query_coords.to(occ.device)

    new_occupancy_gt = torch.zeros(new_query_coords.shape[0], dtype=torch.float32, device=occ.device)
    occ = torch.cat([occ, new_occupancy_gt])
    points = torch.cat([points, new_query_coords])
    return (points, occ)


def sample_voxels_2(voxels: torch.tensor=None, voxel_res: int=256, npoints: List[int]=[25000, 25000], vol_range=[-1.0,1.0], scale=0.8) -> Tuple[torch.Tensor, torch.Tensor]:
    
    n_surface_points, n_in_out_points, n_pad_points = npoints

    kernel = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    ])
    kernel = convert_to_torch_tensor(np_arr=kernel, data_type="fp32", device="cuda")
    kernel = kernel.unsqueeze(0).unsqueeze(0)   
    voxels_f = voxels.unsqueeze(0).unsqueeze(0)  # Add batch & channel dim
    neighbor_sum = F.conv3d(voxels_f, kernel, padding=1).squeeze()

    surf_voxel_coords = torch.nonzero((voxels == 1) & (neighbor_sum < 7), as_tuple=False)
    
    n_surface_voxels = surf_voxel_coords.shape[0]
    sampled_surf_vox_coords = surf_voxel_coords[torch.randperm(n_surface_voxels)[:n_surface_points]]

    if sampled_surf_vox_coords.shape[0] < n_surface_points:
        n_in_out_points += n_surface_points - sampled_surf_vox_coords.shape[0]

    perturbation = (torch.randn_like(sampled_surf_vox_coords, dtype=torch.float32) - 0.5) * 2  # [-1, 1] shift
    near_sampled_surf_vox_coords = sampled_surf_vox_coords + perturbation
    near_sampled_surf_vox_coords = torch.clamp(near_sampled_surf_vox_coords, 0, voxel_res - 1)
    near_sampled_surf_vox_coords = near_sampled_surf_vox_coords.long()

    near_surf_occ = voxels[
        near_sampled_surf_vox_coords[:, 0], 
        near_sampled_surf_vox_coords[:, 1], 
        near_sampled_surf_vox_coords[:, 2]
    ]

    # sample points in range [0,256]
    grid_coords = (torch.rand((n_in_out_points, 3), device=near_sampled_surf_vox_coords.device) * voxel_res).long().clamp(0, voxel_res-1)
    occ_gt_uniform = voxels[grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]]

    occ_gt = torch.cat([near_surf_occ, occ_gt_uniform])

    query_points = torch.cat([near_sampled_surf_vox_coords, grid_coords], dim=0)
    query_points = (query_points.float() / voxel_res) * (vol_range[1] - vol_range[0]) + vol_range[0]

    query_points, occ_gt = pad_volume(points=query_points, occ=occ_gt, nsample=n_pad_points, sample_range=vol_range, pad=0.1)

    return (query_points, occ_gt)


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