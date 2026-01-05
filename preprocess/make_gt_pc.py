import argparse
import os 
import sys
import tqdm
from typing import List
import numpy as np
import subprocess
import open3d as o3d
import torch
from typing import List, Tuple
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from src.utils import read_text, get_split_shape_ids, convert_to_torch_tensor
from src.utils.mesh import read_full_mesh_from_part_json, sample_voxels, sample_voxels_2
from src.utils.binvox_rw import read_as_3d_array


def process_shape_id(shape_id_path: str) -> Tuple[np.array, np.array]:
    point_sample_folder_path = os.path.join(shape_id_path, "point_sample")

    # gt labels        
    gt_seg_labels = np.loadtxt(fname=f"{point_sample_folder_path}/label-10000.txt", dtype=np.int64)
    gt_pc = np.loadtxt(fname=f"{point_sample_folder_path}/pts-10000.txt", dtype=np.float32)

    return (gt_pc, gt_seg_labels)


def main(
        in_root: str, 
        out_root: str, 
        splits=["train", "test"]
    ) -> None:
    cats = read_text(f"{out_root}/cats.txt")
    for cat in cats:
        cat_out_path = os.path.join(out_root, cat)
        cat_in_path =  os.path.join(in_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids(path=f"{cat_out_path}/split.json", split_type=split)
            data = dict()
            data["gt_points"], data["gt_labels"] = [], []
            outfilepath = f"{cat_out_path}/{cat}_{split}_points.hdf5"
            with h5py.File(outfilepath, 'w') as f:
                num_shapes = len(shape_ids)
                gt_points_hdf5 = f.create_dataset("gt_points", shape=(num_shapes, 10000, 3), dtype=np.float32, compression="gzip", chunks=(1, 10000, 3))
                gt_labels_hdf5 = f.create_dataset("gt_labels", shape=(num_shapes, 10000), dtype=np.int64, compression="gzip", chunks=(1, 10000))

                with tqdm.tqdm(total=len(shape_ids), desc=f"Processing {cat}, split {split}") as pbar:
                    for i, shape_id in enumerate(shape_ids):
                        shape_id_path = os.path.join(cat_in_path, shape_id)
                        gt_pc, gt_seg_labels = process_shape_id(shape_id_path=shape_id_path)
                        
                        gt_points_hdf5[i] = gt_pc
                        gt_labels_hdf5[i] = gt_seg_labels
                        
                        pbar.update(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_root", type=str, required=True, help="Path to the data root")
    parser.add_argument("--out_data_root", type=str, required=True, help="Path to the output data root")
    args = parser.parse_args()

    in_data_root = args.in_data_root
    out_data_root = args.out_data_root

    main(
        in_root=in_data_root, 
        out_root=out_data_root,
    )