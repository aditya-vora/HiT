import torch 
import os 
import h5py
import numpy as np
from typing import List, Optional
import json
from torch.utils.data import Dataset

from config import *
from utils import get_split_shape_ids

class ShapeNetDataset(Dataset):
    def __init__(
            self, 
            config: DataConfig, 
            train_mode: bool=True, 
            recon_mode: bool=False, 
            pc_mode: bool=False, 
            mesh_mode: bool=False, 
            iou_mode: bool=False, 
            cd_mode: bool=False
        ) -> None:
        super(ShapeNetDataset, self).__init__()
        """
        This dataset loader loads shapenet data on the fly using lazy loading for all category training.
        """

        self.train_mode = train_mode
        self.recon = recon_mode
        self.pointcloud = pc_mode
        self.mesh = mesh_mode
        self.iou = iou_mode
        self.cd = cd_mode
        self.config = config

        self._data_indexs = []
        self.data_file_paths, self.pc_file_paths, self.mesh_file_paths, self.gt_points_file_paths = {}, {}, {}, {}
        for i, split_name in enumerate(config.splits): 
            for j, cat_name in enumerate(config.cats_list):
                data_file_path = os.path.join(config.data_dir, cat_name, f"{cat_name}_{split_name}_vox.hdf5")
                pc_file_path = os.path.join(config.data_dir, cat_name, f"{cat_name}_{split_name}_pc.hdf5")

                self.data_file_paths[f"{cat_name}_{split_name}"] = h5py.File(data_file_path, 'r')
                self.pc_file_paths[f"{cat_name}_{split_name}"] = h5py.File(pc_file_path, 'r')
                shape_ids = get_split_shape_ids(
                    path=os.path.join(config.data_dir, cat_name, "split.json"), 
                    split_type=split_name
                )

                if mesh_mode:
                    mesh_file_path = os.path.join(config.data_dir, cat_name, f"{cat_name}_{split_name}_meshes.hdf5")
                    self.mesh_file_paths[f"{cat_name}_{split_name}"] = h5py.File(mesh_file_path, 'r')

                if pc_mode or iou_mode:
                    gt_points_file_path = os.path.join(config.data_dir, cat_name, f"{cat_name}_{split_name}_points.hdf5")
                    self.gt_points_file_paths[f"{cat_name}_{split_name}"] = h5py.File(gt_points_file_path, 'r')

                self._data_indexs.extend([(f"{cat_name}_{split_name}", (i, shape_id)) for i, shape_id in enumerate(shape_ids)])

    def __len__(self):
        return len(self._data_indexs)

    def index_of(self, shape_id: str) -> int:
        """Returns the dataset index for a given shape id, for single-shape evaluation."""
        for i, (_, (_, sid)) in enumerate(self._data_indexs):
            if sid == shape_id:
                return i
        raise ValueError(f"shape_id {shape_id} not found in dataset.")

    def __getitem__(self, index):
                
        hdf5_filename, (shape_hdf5_idx, shape_id) = self._data_indexs[index]

        qpts = self.data_file_paths[hdf5_filename]['points'][shape_hdf5_idx]
        qvals = self.data_file_paths[hdf5_filename]['values'][shape_hdf5_idx]
        pc = self.pc_file_paths[hdf5_filename][f'pc'][shape_hdf5_idx]
        surf_scale = self.data_file_paths[hdf5_filename][f'scale'][shape_hdf5_idx][()]

        pc = pc * surf_scale

        if not self.cd:
            pc_indexs = np.random.default_rng().choice(pc.shape[0], self.config.npc, replace=False)
            pc = pc[pc_indexs]

        if not self.cd:
            qpts_indexs = np.random.default_rng().choice(qpts.shape[0], self.config.pts_per_shape, replace=False)
            qpts, qvals = qpts[qpts_indexs], qvals[qpts_indexs]

        data = dict()
        data['querypts'] = qpts.astype(np.float32)
        data['occgt'] = qvals.astype(np.float32)
        data['pc'] = pc.astype(np.float32)
        data['fileid'] = index
        data['shape_id'] = shape_id
        data['category'] = hdf5_filename.split('_')[0]

        if self.pointcloud or self.iou:
            gt_pc = self.gt_points_file_paths[hdf5_filename]['gt_points'][shape_hdf5_idx][:]
            gt_pc_vals = self.gt_points_file_paths[hdf5_filename]['gt_labels'][shape_hdf5_idx][:]
            data['gt_points'] = (gt_pc * surf_scale).astype(np.float32)
            data['gt_points_values'] = gt_pc_vals.astype(np.int64)

        if self.mesh:
            mesh_vertices = self.mesh_file_paths[hdf5_filename][shape_id]['vertices'][:]
            mesh_faces = self.mesh_file_paths[hdf5_filename][shape_id]['faces'][:]
            data['mesh_vertices'] = (mesh_vertices * surf_scale).astype(np.float32)
            data['mesh_faces'] = mesh_faces.astype(np.int32)

        return data