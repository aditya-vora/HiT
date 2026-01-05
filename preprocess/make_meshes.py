import argparse
import os 
import sys
import tqdm
import h5py
import trimesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from src.utils import read_text, get_split_shape_ids

def main(
        in_root: str, 
        out_root: str, 
        splits=["train", "test"],
    ) -> None:
    cats = read_text(f"{out_root}/cats.txt")
    for cat in cats:
        cat_out_path = os.path.join(out_root, cat)
        cat_in_path =  os.path.join(in_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids(path=f"{cat_out_path}/split.json", split_type=split)

            outfilepath = f"{cat_out_path}/{cat}_{split}_meshes.hdf5"
            with h5py.File(outfilepath, 'w') as f:
                with tqdm.tqdm(total=len(shape_ids), desc=f"Processing {cat}, split {split}") as pbar:
                    for i, shape_id in enumerate(shape_ids):
                        mesh_group = f.create_group(f"{shape_id}")
                        shape_id_path = os.path.join(cat_in_path, shape_id) 
                        mesh_path = os.path.join(shape_id_path, "fullmesh_norm.obj")
                        mesh_obj = trimesh.load(mesh_path, force="mesh", process=False)
                        mesh_group.create_dataset("vertices", data=mesh_obj.vertices, compression="gzip", dtype="float32")
                        mesh_group.create_dataset("faces", data=mesh_obj.faces, compression="gzip", dtype="int32")                        
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