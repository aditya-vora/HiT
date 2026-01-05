import argparse
import os 
import tqdm
import numpy as np
import h5py

from utils import read_text, get_split_shape_ids
from utils.binvox_rw import read_as_3d_array

def main(
        in_root: str, 
        out_root: str, 
        gridsize: int=512,
        npoints: int=50000,
        splits=["train", "test"]
    ) -> None:
    cats = read_text(f"{out_root}/cats.txt")
    for cat in cats:
        cat_out_path = os.path.join(out_root, cat)
        cat_in_path =  os.path.join(in_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids(path=f"{cat_out_path}/split.json", split_type=split)

            data = dict()
            data["points"], data["values"], data["voxels"] = [], [], []

            outfilepath = f"{cat_out_path}/{cat}_{split}_vox.hdf5"

            with h5py.File(outfilepath, 'w') as f:

                num_shapes = len(shape_ids)

                query_pts_hdf5 = f.create_dataset("points", shape=(num_shapes, npoints, 3), dtype=np.float32, compression="gzip", chunks=(1, npoints, 3))
                occ_pts_hdf5 = f.create_dataset("values", shape=(num_shapes, npoints, 1), dtype=np.bool_, compression="gzip", chunks=(1, npoints, 1))
                vox_pts_hdf5 = f.create_dataset("voxels", shape=(num_shapes, gridsize, gridsize, gridsize), dtype=np.bool_, compression="gzip", chunks=(1, gridsize, gridsize, gridsize))

                with tqdm.tqdm(total=len(shape_ids), desc=f"Processing {cat}, split {split}") as pbar:
                    for i, shape_id in enumerate(shape_ids):
                        shape_id_path = os.path.join(cat_in_path, shape_id)
                        
                        query_pts_path = os.path.join(shape_id_path, "query_pts.npy" )
                        occ_gt_path = os.path.join(shape_id_path, "occ_gt.npy")
                        voxel_path = os.path.join(shape_id_path, "voxelized", f"fullmesh_norm.obj_{gridsize}_solid.binvox")

                        with open(voxel_path, "rb") as f:
                            voxels = read_as_3d_array(f)
                        
                        voxels = voxels.data.astype(np.bool_) 
                        query_pts = np.load(query_pts_path)
                        occ_gt = np.load(occ_gt_path).astype(np.bool_)[..., None]

                        query_pts_hdf5[i] = query_pts
                        occ_pts_hdf5[i] = occ_gt
                        vox_pts_hdf5[i] = voxels

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
        npoints=60000
    )