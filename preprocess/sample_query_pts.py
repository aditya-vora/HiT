import argparse
import os 
import sys
import tqdm
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils import read_text, get_split_shape_ids, convert_to_torch_tensor
from utils.mesh import sample_voxels_2
from utils.binvox_rw import read_as_3d_array

def process_shape_id(
        shape_id_path: str, 
        grid_size: int=512, 
        npoints=[25000,25000]
    ) -> np.array:

    binvox_filepath = os.path.join(shape_id_path, "voxelized", f"fullmesh_norm.obj_{grid_size}_solid.binvox")
    
    with open(binvox_filepath, "rb") as f:
        voxels = read_as_3d_array(f)
    
    voxels = voxels.data.astype(np.float32) 
    voxels = convert_to_torch_tensor(np_arr=voxels, data_type=torch.float32, device="cuda")

    query_pts, occ_gt = sample_voxels_2(
        voxels=voxels, 
        voxel_res=grid_size, 
        npoints=npoints,
        vol_range=[-0.5, 0.5],
        scale=1.0
    )

    query_pts = query_pts.cpu().numpy()
    occ_gt = occ_gt.cpu().numpy()

    query_pts_path = os.path.join(shape_id_path, "query_pts.npy")
    occ_gt_path = os.path.join(shape_id_path, "occ_gt.npy")

    np.save(query_pts_path, query_pts)
    np.save(occ_gt_path, occ_gt)


def main(
        in_root: str, 
        out_root: str, 
        gridsize: int=512,
        splits=["train", "test"]
    ) -> None:
    cats = read_text(f"{out_root}/cats.txt")
    for cat in cats:
        cat_out_path = os.path.join(out_root, cat)
        cat_in_path =  os.path.join(in_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids(
                path=f"{cat_out_path}/split.json", 
                split_type=split
            )

            with tqdm.tqdm(total=len(shape_ids), desc=f"Processing {cat}, split {split}") as pbar:
                for shape_id in shape_ids:
                    shape_id_path = os.path.join(cat_in_path, shape_id)
                    process_shape_id(
                        shape_id_path=shape_id_path,
                        grid_size=gridsize,
                        npoints=[24000,24000,12000]
                    )
                    pbar.update(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_root", type=str, required=True, help="Path to the data root")
    parser.add_argument("--out_data_root", type=str, required=True, help="Path to the output data root")
    parser.add_argument("--gridsize", type=int, default=256, help="Voxel grid size")    
    args = parser.parse_args()

    in_data_root = args.in_data_root
    out_data_root = args.out_data_root
    grid_size = args.gridsize

    main(
        in_root=in_data_root, 
        out_root=out_data_root,
        gridsize=grid_size,
    )