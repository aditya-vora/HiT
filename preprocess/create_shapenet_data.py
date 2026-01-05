"""This script creates train/val/test splits for the dataset.
It reads the shape IDs from the provided split files and verifies the existence of required data files.
It then saves the splits in a JSON format for later use.
"""

import argparse 
import sys 
import os
import json
from typing import List
import tqdm
import numpy as np
import h5py
import open3d as o3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_text, get_split_shape_ids_from_text_file, get_split_shape_ids
from utils.mesh import read_full_mesh_from_part_json

def process_shape_id(shape_id_path: str, npoints: List[int]) -> np.array:
    obj_files_root = os.path.join(shape_id_path, "objs")                
 
    # read the part json file
    mesh_obj = read_full_mesh_from_part_json(
        json_path=f"{shape_id_path}/result.json", 
        obj_root_path=obj_files_root
    )

    mesh_obj = scale_mesh_unit_sphere(mesh=mesh_obj)
    # sample the point cloud from the mesh
    vertices, faces = mesh_obj.vertices, mesh_obj.faces 
    vertices = convert_to_torch_tensor(np_arr=vertices, data_type="fp32")
    faces = convert_to_torch_tensor(np_arr=faces, data_type="int64")

    # sample the point cloud
    pc, f_indxs, normals = sample_surface_torch(
        faces=faces,
        vs=vertices.unsqueeze(0), 
        count=npoints
    )
    pc = convert_to_np_array(pc)    
    return pc

def create_splits_from_lst(in_root: str, out_root: str, process: str="02691156", splits: List[str]=["train", "val", "test"]) -> None:  
    """function which creates splits from lst files
    Args:
        in_root (str): _path to the input data root_
        out_root (str): _path to the output data root_
        process (str, optional): category to process. Defaults to "02691156".
        splits (List[str], optional): splits to process. Defaults to ["train", "val", "test"].

    Returns:
        None: _saves the splits in json format_    
    """
    # Read the splits
    cats = read_text(f"{out_root}/cats.txt")
    for cat in cats:
        split_out_path = os.path.join(out_root, cat)
        os.makedirs(split_out_path, exist_ok=True)
        split_list = dict()
        cat_in_path = os.path.join(in_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids_from_text_file(path=os.path.join(cat_in_path, f"{split}.lst"))
            with tqdm.tqdm(total=len(shape_ids), desc=f"Processing {cat}") as pbar:
                for i, shape_id in enumerate(shape_ids):
                    qpath = os.path.join(cat_in_path, "3_query_points", f"{shape_id}.npz")    
                    ppath = os.path.join(cat_in_path, "4_pointcloud", f"{shape_id}.npz") 
                    if not os.path.exists(qpath) or not os.path.exists(ppath):
                        print(f"Shape {shape_id} does not exist")
                        shape_ids.remove(shape_id)
                    pbar.update(1)

                split_list[split] = shape_ids
        # Save the splits
        json.dump(split_list, open(f"{split_out_path}/split.json", 'w'))


def sample_pc(in_root: str, out_root: str, splits: List[str]=["train"], n_points: List[int]=[1024, 2048]) -> None:
    cats = read_text(f"{out_root}/cats.txt")
    for cat in cats:
        cat_out_path = os.path.join(out_root, cat)
        cat_in_path =  os.path.join(in_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids(path=f"{cat_out_path}/split.json", split_type=split)
            # save the point clouds
            out_filepath = f"{cat_out_path}/{cat}_{split}_pc.hdf5"            
            with h5py.File(out_filepath, 'w') as f:
                for npts in n_points:
                    num_shapes = len(shape_ids)
                    
                    pts_hdf5 = f.create_dataset(
                        f"pc{npts}", 
                        shape=(num_shapes, npts, 3), 
                        dtype=np.float32, 
                        compression="gzip", 
                        chunks=(1, npts, 3)
                    )

                    with tqdm.tqdm(total=len(shape_ids), desc=f"Processing {cat}, split {split} with {npts} pts") as pbar:
                        for i, shape_id in enumerate(shape_ids):
                            shape_id_path = os.path.join(cat_in_path, shape_id)
                            pc = process_shape_id(shape_id_path=shape_id_path, npoints=npts)
                            # Save the point cloud to a PLY file
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc)
                            ply_filepath = os.path.join(shape_id_path, f"pc_{npts}.ply")
                            o3d.io.write_point_cloud(ply_filepath, pcd)
                            pts_hdf5[i] = pc
                            pbar.update(1)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_root", type=str, default="", help="Path to the data root")
    parser.add_argument("--data_out_root", type=str, default="", help="Path to the output data root")
    parser.add_argument("--process", type=str, default="02691156", help="Categories to process")    
    args = parser.parse_args()

    in_data_root = args.raw_data_root   
    out_data_root = args.data_out_root
    process = args.process

    create_splits_from_lst(in_root=in_data_root, out_root=out_data_root, process=process)