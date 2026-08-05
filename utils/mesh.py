import os
import trimesh
import mcubes
from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance, cKDTree
from typing import Tuple, List
from scipy.ndimage import binary_erosion, binary_dilation

from utils import read_json, convert_to_torch_tensor, convert_to_np_array, COLOR_LIST


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


def assign_part_labels(points: torch.Tensor, part_occ: torch.Tensor, smooth: bool = True) -> np.array:
    """Assigns each point to its most likely part via argmax over per-part occupancy.

    Args:
        points (torch.Tensor): query points, [1, N, 3] or [N, 3].
        part_occ (torch.Tensor): per-part occupancy at each point, [1, N, P] or [N, P].
        smooth (bool): if True, propagate labels from confidently-assigned points to
            ambiguous ones via nearest-neighbor lookup.
    Returns:
        np.array: part label per point, [N].
    """
    if points.dim() == 3:
        points = points.squeeze(0)
    if part_occ.dim() == 3:
        part_occ = part_occ.squeeze(0)

    points_np = convert_to_np_array(tensor=points)
    occ_np = convert_to_np_array(tensor=part_occ)
    labels = np.argmax(occ_np, axis=-1)

    if not smooth:
        return labels

    confident = np.max(occ_np, axis=-1) > 1e-2
    if not np.any(confident):
        return labels

    tree = cKDTree(points_np[confident])
    _, nearest_idx = tree.query(points_np)
    return labels[confident][nearest_idx]


def occupancy_grid_to_mesh(occ: torch.Tensor, density: int, mcubeth: float, interval: List[float] = [-1.0, 1.0]) -> trimesh.Trimesh:
    """Runs marching cubes on an occupancy field evaluated on a uniform density^3 grid.

    Args:
        occ (torch.Tensor): occupancy values on the grid, [density**3].
        density (int): number of samples per axis of the evaluation grid.
        mcubeth (float): marching cubes iso-surface threshold.
        interval (List[float]): [min, max] world-space extent the grid was sampled over,
            used to rescale marching-cubes' grid-index vertices back to world coordinates.
    Returns:
        trimesh.Trimesh: extracted mesh, in world coordinates.
    """
    volume = occ.view(density, density, density).permute(1, 0, 2).cpu().detach().numpy()
    verts, faces = mcubes.marching_cubes(volume, mcubeth)
    verts = verts / (density - 1) * (interval[1] - interval[0]) + interval[0]
    return trimesh.Trimesh(verts, faces)


def hierarchical_occupancy_to_meshes(
        shape_occ: torch.Tensor,
        part_occ: Dict[str, torch.Tensor],
        density: int,
        mcubeth: float,
        interval: List[float] = [-1.0, 1.0],
    ) -> Tuple[List[trimesh.Trimesh], Dict[str, List[trimesh.Trimesh]], Dict[str, List[int]]]:
    """Extracts the full-shape mesh and a per-part mesh at each hierarchy level via marching cubes.

    Args:
        shape_occ (torch.Tensor): full-shape occupancy on a density^3 grid, [density**3].
        part_occ (Dict[str, torch.Tensor]): per-level per-part occupancy, {"level_i": [density**3, n_parts_i]}.
        density (int): number of samples per axis of the evaluation grid.
        mcubeth (float): marching cubes iso-surface threshold.
        interval (List[float]): [min, max] world-space extent the grid was sampled over.
    Returns:
        Tuple[List[trimesh.Trimesh], Dict[str, List[trimesh.Trimesh]], Dict[str, List[int]]]:
        the full-shape mesh (single-element list), the per-level list of part meshes, and their part labels.
    """
    shape_mesh = [occupancy_grid_to_mesh(occ=shape_occ, density=density, mcubeth=mcubeth, interval=interval)]

    part_meshes, part_labels = {}, {}
    for level, occ in part_occ.items():
        volume = occ.view(density, density, density, -1).permute(1, 0, 2, 3).cpu().detach().numpy()
        level_meshes, level_labels = [], []
        for part_id in range(occ.shape[-1]):
            verts, faces = mcubes.marching_cubes(volume[..., part_id], mcubeth)
            verts = verts / (density - 1) * (interval[1] - interval[0]) + interval[0]
            level_meshes.append(trimesh.Trimesh(verts, faces))
            level_labels.append(part_id)
        part_meshes[level] = level_meshes
        part_labels[level] = level_labels

    return shape_mesh, part_meshes, part_labels


def write_colored_obj(vertices: np.array, faces: np.array, colors: List[str], filepath: str) -> None:
    """Writes a mesh to .obj with a "r g b" color (in [0, 1]) appended to each vertex line."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as fout:
        for vertex, color in zip(vertices, colors):
            fout.write(f"v {vertex[0]} {vertex[1]} {vertex[2]} {color}\n")
        for face in faces:
            fout.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def write_part_meshes(
        part_meshes: Dict[str, List[trimesh.Trimesh]],
        part_labels: Dict[str, List[int]],
        shape_mesh: List[trimesh.Trimesh],
        out_dir: str,
    ) -> None:
    """Writes a color-coded, per-part mesh for each hierarchy level, plus the full shape mesh, as .obj files.

    Args:
        part_meshes (Dict[str, List[trimesh.Trimesh]]): per-level list of part meshes.
        part_labels (Dict[str, List[int]]): per-level list of part labels, aligned with part_meshes.
        shape_mesh (List[trimesh.Trimesh]): full-shape mesh (single-element list), or None.
        out_dir (str): output directory.
    Returns:
        None
    """
    os.makedirs(out_dir, exist_ok=True)

    for level, meshes in part_meshes.items():
        vertices, faces, colors = [], [], []
        vertex_offset = 0
        for mesh, label in zip(meshes, part_labels[level]):
            if mesh.vertices.shape[0] == 0:
                continue
            color = [c / 255.0 for c in map(int, COLOR_LIST[label % len(COLOR_LIST)].split(" "))]
            color_str = " ".join(map(str, color))
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + vertex_offset)
            colors.extend([color_str] * len(mesh.vertices))
            vertex_offset += len(mesh.vertices)

        if len(vertices) == 0:
            continue

        write_colored_obj(
            vertices=np.concatenate(vertices, axis=0),
            faces=np.concatenate(faces, axis=0),
            colors=colors,
            filepath=os.path.join(out_dir, f"parts_{level}.obj"),
        )

    if shape_mesh is not None and len(shape_mesh) > 0 and shape_mesh[0].vertices.shape[0] > 0:
        shape_mesh[0].export(os.path.join(out_dir, "full_mesh.obj"))


def save_labeled_point_cloud(points: np.array, labels: np.array, filepath: str) -> None:
    """Writes an ASCII .ply point cloud, colored per-point by an integer part label.

    Args:
        points (np.array): point coordinates, [N, 3].
        labels (np.array): part label per point, [N].
        filepath (str): output .ply filepath.
    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n_points = points.shape[0]
    with open(filepath, 'w') as fout:
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write(f"element vertex {n_points}\n")
        fout.write("property float x\nproperty float y\nproperty float z\n")
        fout.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fout.write("end_header\n")
        for point, label in zip(points, labels):
            color = COLOR_LIST[int(label) % len(COLOR_LIST)]
            fout.write(f"{point[0]} {point[1]} {point[2]} {color}\n")