import torch 
import os 
import h5py
import numpy as np
from typing import List, Optional
import json
from torch.utils.data import Dataset

from config import Config
from utils import get_split_shape_ids

class ShapeNetDataset(Dataset):
    def __init__(self, config: Config) -> None:
        super(ShapeNetDataset, self).__init__()
        """
        This dataset loader loads shapenet data on the fly using lazy loading for all category training.
        """
        # self.data_dir = config.data_dir
        # self.pointcloud = pointcloud
        # self.iou = iou
        # self.mesh = mesh 
        # self.cd = cd
        # self.cats = config.cats_list
        # self.splits = split
        # self.n_pc = n_pc
        # self.n_qpts = n_qpts

        self._data_indexs = [] 
        self.data_file_paths, self.vox_file_paths, self.pc_file_paths, self.mesh_file_paths, self.gt_points_file_paths = {}, {}, {}, {}, {}
        for i, split in enumerate(config.splits): 
            for j, cat in enumerate(config.cats_list):
                data_file_path = os.path.join(config.data_dir, cat, f"{cat}_{split}_vox.hdf5")
                # vox_file_path = os.path.join(self.data_dir, cat, f"{cat}_{split}_vox_down.hdf5")
                pc_file_path = os.path.join(config.data_dir, cat, f"{cat}_{split}_pc.hdf5")
                
                self.data_file_paths[f"{cat}_{split}"] = h5py.File(data_file_path, 'r')
                # self.vox_file_paths[f"{cat}_{split}"] = h5py.File(vox_file_path, 'r')
                self.pc_file_paths[f"{cat}_{split}"] = h5py.File(pc_file_path, 'r')
                shape_ids = get_split_shape_ids(path=os.path.join(config.data_dir, cat, "split.json"), split_type=split)

                if self.mesh:
                    mesh_file_path = os.path.join(self.data_dir, cat, f"{cat}_{split}_meshes.hdf5")
                    self.mesh_file_paths[f"{cat}_{split}"] = h5py.File(mesh_file_path, 'r')

                if self.pointcloud or self.iou:
                    gt_points_file_path = os.path.join(self.data_dir, cat, f"{cat}_{split}_points.hdf5")
                    self.gt_points_file_paths[f"{cat}_{split}"] = h5py.File(gt_points_file_path, 'r')

                self._data_indexs.extend([(f"{cat}_{split}", (i, shape_id)) for i, shape_id in enumerate(shape_ids)])

    def __len__(self):
        return len(self._data_indexs)

    def __getitem__(self, index):
                
        hdf5_filename, (shape_hdf5_idx, shape_id) = self._data_indexs[index]

        qpts = self.data_file_paths[hdf5_filename]['points'][shape_hdf5_idx]
        qvals = self.data_file_paths[hdf5_filename]['values'][shape_hdf5_idx]
        pc = self.pc_file_paths[hdf5_filename][f'pc'][shape_hdf5_idx]
        # surf_scale = self.pc_file_paths[hdf5_filename][f'scale'][shape_hdf5_idx][()]
        surf_scale = self.data_file_paths[hdf5_filename][f'scale'][shape_hdf5_idx][()]

        pc = pc * surf_scale

        if not self.cd: 
            pc_indexs = np.random.default_rng().choice(pc.shape[0], self.n_pc, replace=False)
            pc = pc[pc_indexs]
        else:
            pc = pc


        if not self.cd:
            # voxels = self.vox_file_paths[hdf5_filename][f'voxels_{self.vox_res}'][shape_hdf5_idx][:]
            qpts_indexs = np.random.default_rng().choice(qpts.shape[0], self.n_qpts, replace=False) 
            qpts, qvals = qpts[qpts_indexs], qvals[qpts_indexs]
        else:
            qpts, qvals = qpts, qvals

        
        data = dict() 
        data['querypts'] = qpts.astype(np.float32) 
        data['occgt'] = qvals.astype(np.float32)
        data['pc'] = pc.astype(np.float32) 
        data['fileid'] = index        
        # data['query_idxs'] = qpts_indexs
        # data['pc_idxs'] = pc_indexs
        # data['voxels'] = voxels.astype(np.float32) 
        data['shape_id'] = shape_id
        data['category'] = hdf5_filename.split('_')[0]


        
        # if self.pointcloud or self.iou:
        #     gt_pc = self.gt_points_file_paths[hdf5_filename]['gt_points'][shape_hdf5_idx][:]
        #     gt_pc_vals = self.gt_points_file_paths[hdf5_filename]['gt_labels'][shape_hdf5_idx][:]
        #     min_gt_pc, max_gt_pc = np.min(gt_pc, axis=0), np.max(gt_pc, axis=0)
        #     d3 = np.linalg.norm(max_gt_pc - min_gt_pc)
        #     scale_gt = d2/d3
        #     data['gt_points'] = gt_pc.astype(np.float32) * scale_gt
        #     data['gt_points_values'] = gt_pc_vals.astype(np.float32)

        # if self.mesh: 
        #     mesh_vertices = self.mesh_file_paths[hdf5_filename][shape_id]['vertices'][:]
        #     mesh_faces = self.mesh_file_paths[hdf5_filename][shape_id]['faces'][:]
        #     min_verts, max_verts = np.min(mesh_vertices, axis=0), np.max(mesh_vertices, axis=0)
        #     d4 = np.linalg.norm(max_verts - min_verts)
        #     scale_mesh = d2/d4
        #     data['mesh_vertices'] = mesh_vertices.astype(np.float32) * scale_mesh
        #     data['mesh_faces'] = mesh_faces.astype(np.int32)

        return data