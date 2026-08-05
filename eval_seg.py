import os
import numpy as np
import torch
import trimesh

from config import *
from utils import read_text, setup_experiment_directory, convert_to_torch_tensor, load_checkpoint
from utils.logger import create_logger
from utils.mesh import (
    assign_part_labels,
    occupancy_grid_to_mesh,
    hierarchical_occupancy_to_meshes,
    write_part_meshes,
    save_labeled_point_cloud,
)
from utils.model import reconstruct_on_grid
from utils.metrics import segmentation_miou, chamfer_distance_and_fscore
from src.models.hit import HiT_models
from src.dataset.build import build_dataset


#################################################################################
#                                Model Loading                                  #
#################################################################################
def load_model(model_config: ModelConfig, ckpt: str, device: torch.device) -> torch.nn.Module:
    model = HiT_models[model_config.model_type](
        pf_dim=model_config.pf_dim,
        zf_dim=model_config.zf_dim,
        tf_dim=model_config.tf_dim,
        ef_dim=model_config.ef_dim,
        gf_dim=model_config.gf_dim,
        n_parts=model_config.n_parts,
        n_levels=model_config.n_levels,
        n_planes=model_config.n_planes,
        planef_dim=model_config.planef_dim,
        mask_mode=model_config.mask_mode
    ).to(device=device)

    assert ckpt != "", "a checkpoint path must be provided via --ckpt for evaluation."
    load_checkpoint(model=model, filename=ckpt)
    model.eval()
    return model


#################################################################################
#                          Reconstruction / Mesh Export                         #
#################################################################################
def reconstruct_shape(model: torch.nn.Module, data: dict, eval_config: EvalConfig, device: torch.device, out_dir: str) -> None:
    pc = convert_to_torch_tensor(np_arr=data['pc'], data_type="fp32", device=device, batchify=True)

    shape_occ, part_occ = reconstruct_on_grid(
        model=model, pc=pc, density=eval_config.density, nchunks=eval_config.nchunks, interval=eval_config.interval
    )
    shape_mesh, part_meshes, part_labels = hierarchical_occupancy_to_meshes(
        shape_occ=shape_occ, part_occ=part_occ, density=eval_config.density, mcubeth=eval_config.mcubeth, interval=eval_config.interval
    )
    write_part_meshes(part_meshes=part_meshes, part_labels=part_labels, shape_mesh=shape_mesh, out_dir=out_dir)


def run_reconstruction(model, dataset, data_config: DataConfig, eval_config: EvalConfig, device, out_dir: str, logger) -> None:
    shape_index = dataset.index_of(data_config.shape_id)
    data = dataset[shape_index]
    reconstruct_shape(model=model, data=data, eval_config=eval_config, device=device, out_dir=os.path.join(out_dir, data['shape_id']))
    logger.info(f"saved reconstruction for shape {data['shape_id']} to {out_dir}")


def run_mesh_export(model, dataset, eval_config: EvalConfig, device, out_dir: str, logger) -> None:
    for i in range(len(dataset)):
        data = dataset[i]
        shape_out_dir = os.path.join(out_dir, data['category'], data['shape_id'])
        reconstruct_shape(model=model, data=data, eval_config=eval_config, device=device, out_dir=shape_out_dir)
        logger.info(f"[{i + 1}/{len(dataset)}] saved mesh for shape {data['shape_id']} ({data['category']})")


#################################################################################
#                    Segmented Point Cloud / Segmentation IoU                   #
#################################################################################
def predict_part_labels(model: torch.nn.Module, data: dict, eval_config: EvalConfig, device: torch.device) -> np.array:
    pc = convert_to_torch_tensor(np_arr=data['pc'], data_type="fp32", device=device, batchify=True)
    gt_points = convert_to_torch_tensor(np_arr=data['gt_points'], data_type="fp32", device=device, batchify=True)

    feats = model.encoder(pc)
    pred = model.decoder(gt_points, pc, feats, enc_type="volume")
    finest_level = f"level_{pred['num_active_blocks'] - 1}"
    part_occ = pred['all_convex_part_indicator_hat'][finest_level]

    return assign_part_labels(points=gt_points, part_occ=part_occ, smooth=eval_config.use_post_processing)


@torch.no_grad()
def run_pointcloud_export(model, dataset, eval_config: EvalConfig, device, out_dir: str, logger) -> None:
    for i in range(len(dataset)):
        data = dataset[i]
        pred_labels = predict_part_labels(model=model, data=data, eval_config=eval_config, device=device)

        filepath = os.path.join(out_dir, data['category'], f"{data['shape_id']}.ply")
        save_labeled_point_cloud(points=data['gt_points'], labels=pred_labels, filepath=filepath)
        logger.info(f"[{i + 1}/{len(dataset)}] saved labeled point cloud for shape {data['shape_id']} ({data['category']})")


@torch.no_grad()
def run_iou_eval(model, dataset, eval_config: EvalConfig, device, out_dir: str, logger) -> None:
    os.makedirs(out_dir, exist_ok=True)
    category_ious = {}

    for i in range(len(dataset)):
        data = dataset[i]
        pred_labels = predict_part_labels(model=model, data=data, eval_config=eval_config, device=device)
        shape_miou = segmentation_miou(pred_labels=pred_labels, gt_labels=data['gt_points_values'])

        category_ious.setdefault(data['category'], []).append(shape_miou)
        logger.info(f"[{i + 1}/{len(dataset)}] shape {data['shape_id']} ({data['category']}) mIoU: {shape_miou * 100.0:.1f}")

    with open(os.path.join(out_dir, "iou.txt"), "w") as fout:
        all_ious = []
        for category, ious in category_ious.items():
            category_miou = float(np.mean(ious)) * 100.0
            all_ious.extend(ious)
            fout.write(f"{category}, mIoU: {category_miou:.1f}\n")
            logger.info(f"category {category} mIoU: {category_miou:.1f}")

        overall_miou = float(np.mean(all_ious)) * 100.0
        fout.write(f"overall, mIoU: {overall_miou:.1f}\n")
        logger.info(f"overall mIoU: {overall_miou:.1f}")


#################################################################################
#                                Chamfer Distance                               #
#################################################################################
@torch.no_grad()
def run_cd_eval(model, dataset, eval_config: EvalConfig, device, out_dir: str, logger) -> None:
    os.makedirs(out_dir, exist_ok=True)
    category_cds, category_fscores = {}, {}

    for i in range(len(dataset)):
        data = dataset[i]
        pc = convert_to_torch_tensor(np_arr=data['pc'], data_type="fp32", device=device, batchify=True)

        shape_occ, _ = reconstruct_on_grid(
            model=model, pc=pc, density=eval_config.density, nchunks=eval_config.nchunks, interval=eval_config.interval
        )
        shape_mesh = occupancy_grid_to_mesh(occ=shape_occ, density=eval_config.density, mcubeth=eval_config.mcubeth, interval=eval_config.interval)

        if shape_mesh.vertices.shape[0] == 0:
            logger.info(f"[{i + 1}/{len(dataset)}] shape {data['shape_id']} produced an empty mesh, skipping.")
            continue

        pred_points, _ = trimesh.sample.sample_surface(shape_mesh, eval_config.n_surface_samples)
        cd, fscore = chamfer_distance_and_fscore(
            pred_points=np.asarray(pred_points), gt_points=data['pc'], threshold=eval_config.fscore_threshold
        )

        category_cds.setdefault(data['category'], []).append(cd)
        category_fscores.setdefault(data['category'], []).append(fscore)
        logger.info(f"[{i + 1}/{len(dataset)}] shape {data['shape_id']} ({data['category']}) CD: {cd:.5f}, F-score: {fscore:.3f}")

    with open(os.path.join(out_dir, "cd.txt"), "w") as fout:
        all_cds, all_fscores = [], []
        for category in category_cds:
            category_cd = float(np.mean(category_cds[category]))
            category_fscore = float(np.mean(category_fscores[category]))
            all_cds.extend(category_cds[category])
            all_fscores.extend(category_fscores[category])
            fout.write(f"{category}, CD: {category_cd:.5f}, F-score: {category_fscore:.3f}\n")
            logger.info(f"category {category} CD: {category_cd:.5f}, F-score: {category_fscore:.3f}")

        fout.write(f"overall, CD: {float(np.mean(all_cds)):.5f}, F-score: {float(np.mean(all_fscores)):.3f}\n")
        logger.info(f"overall CD: {float(np.mean(all_cds)):.5f}, F-score: {float(np.mean(all_fscores)):.3f}")


#################################################################################
#                                    Entry Point                                 #
#################################################################################
def main():
    data_config, model_config, train_config, loss_config, eval_config = parse_args()

    assert torch.cuda.is_available(), "Evaluation currently requires at least one GPU."
    torch.cuda.set_device(eval_config.gpu_id)
    device = torch.device(f"cuda:{eval_config.gpu_id}")

    experiment_dir = setup_experiment_directory(base_dir=train_config.exp_dir, exp_name=train_config.exp_name)
    logger = create_logger(experiment_dir, filename="eval_log", mode="single")
    logger.info(f"Evaluation outputs will be saved to {experiment_dir}")

    if data_config.cats == "all":
        data_config.cats_list = read_text(f"{data_config.data_dir}/cats.txt")
    else:
        data_config.cats_list = [data_config.cats]

    model = load_model(model_config=model_config, ckpt=train_config.ckpt, device=device)

    dataset = build_dataset(
        config=data_config,
        train_mode=False,
        recon_mode=model_config.recon_mode,
        pc_mode=model_config.pc_mode,
        mesh_mode=model_config.mesh_mode,
        iou_mode=model_config.iou_mode,
        cd_mode=model_config.cd_mode,
    )
    logger.info(f"Read from: {data_config.data_dir}, Dataset contains {len(dataset):,} shapes.")

    if model_config.recon_mode:
        run_reconstruction(model, dataset, data_config, eval_config, device, out_dir=os.path.join(experiment_dir, "recon"), logger=logger)
    elif model_config.mesh_mode:
        run_mesh_export(model, dataset, eval_config, device, out_dir=os.path.join(experiment_dir, "mesh"), logger=logger)
    elif model_config.pc_mode:
        run_pointcloud_export(model, dataset, eval_config, device, out_dir=os.path.join(experiment_dir, "pointcloud"), logger=logger)
    elif model_config.iou_mode:
        run_iou_eval(model, dataset, eval_config, device, out_dir=os.path.join(experiment_dir, "metrics"), logger=logger)
    elif model_config.cd_mode:
        run_cd_eval(model, dataset, eval_config, device, out_dir=os.path.join(experiment_dir, "metrics"), logger=logger)
    else:
        raise ValueError("no evaluation mode selected; pass one of --recon, --mesh, --pointcloud, --iou, --cd.")


if __name__ == "__main__":
    main()
