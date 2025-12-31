import argparse 
from typing import List
from dataclasses import dataclass, field

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch HiT Training")
    
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for Adam optimizer [0.0001]")
    parser.add_argument("--global_batch_size", type=int, default=24, help="Batch size")
    parser.add_argument("--pts-per-shape", type=int, default=4096, help="Number of points per shape.")
    parser.add_argument("--npc", type=int, default=2048, help="Number of points in the point cloud.")
    parser.add_argument("--exp-dir", type=str, default="/mnt/data/ava/hit/per_cat_exp", help="Directory for saving experiments")
    parser.add_argument("--exp-name", type=str, default="sample", help="Directory for saving experiments")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")

    # parser.add_argument("--data_dir", type=str, default="/mnt/data/ava/hit/data/objaverse", help="Root directory of dataset")
    # parser.add_argument("--dataset_name", type=str, default="objaverse", help="Name of the dataset")
    # parser.add_argument("--cat", type=str, default="objaverse", help="Category of the object")
    # parser.add_argument("--shape_id", type=str, default="1ab3abb5c090d9b68e940c4e64a94e1e", help="Shape ID of the object")

    # parser.add_argument("--train", action="store_true", help="Enable training mode")
    # parser.add_argument("--recon", action="store_true", help="Output reconstructed shape with segmentation")
    # parser.add_argument("--pointcloud", action="store_true", help="Output point cloud with segmentation")
    # parser.add_argument("--mesh", action="store_true", help="Output mesh with segmentation")
    # parser.add_argument("--iou", action="store_true", help="Output IOU for test shapes")
    # parser.add_argument("--cd", action="store_true", help="Output chamfer distance for test shapes")
    # parser.add_argument("--finetune", action="store_true", help="Enable finetuning mode")   
    # parser.add_argument("--vis_cvx", action="store_true", help="Enable visualization of convex hulls")
    # parser.add_argument("--finetune_cat", action="store_true", help="Category for finetuning")
    # parser.add_argument("--vis_attn", action="store_true", help="Enable attention visualization")

    # parser.add_argument("--gpu_id", type=int, default=0, help="GPU to use")
    # parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")

    # parser.add_argument("--log_data", type=int, default=100, help="Directory for saving log data")
    # parser.add_argument("--log_model", type=int, default=1000, help="Directory for saving log model")
    # parser.add_argument("--check_val_every_n_epoch", type=int, default=10, help="Check validation every n epochs")
    parser.add_argument("--nparts", type=int, nargs='+', default=[8,16,36], help="Number of parts")
    # parser.add_argument("--prog_interval", type=List[int], default=[2,4,8,12], help="Interval for progress logging")
    parser.add_argument("--nlevels", type=int, default=3, help="Number of levels")
    parser.add_argument("--nplanes", type=int, default=32, help="Number of planes")
    parser.add_argument("--plane-fdim", type=int, default=8, help="Plane feature dimension")
    parser.add_argument("--pf-dim", type=int, default=3, help="Input dimension")
    parser.add_argument("--zf-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--tf-dim", type=int, default=4, help="Token dimension")
    parser.add_argument("--ef-dim", type=int, default=256, help="Encoder dimension")
    parser.add_argument("--gf-dim", type=int, default=256, help="generator dimension")
    parser.add_argument("--model-type", type=str, default="hit-volume", help="Model name")
    # parser.add_argument("--density", type=int, default=32, help="Density of the point cloud")
    # parser.add_argument("--nchunks", type=int, default=8, help="Number of chunks")
    # parser.add_argument("--mcubeth", type=int, default=0.5, help="Cube size")
    # parser.add_argument("--interval", type=List[float], default=[-1,1] , help="Interval")

    # # loss params 
    # parser.add_argument("--w_im_sample_loss", type=float, default=1.0, help="Weight for MLP occ decoder loss.")
    # parser.add_argument("--w_sample_loss", type=float, default=1.0, help="Weight for convex sample loss.")
    # parser.add_argument("--w_equilibrium_loss", type=float, default=0.001, help="Weight for equilibrium loss.")
    # parser.add_argument("--w_bbx_loss", type=float, default=0.01, help="Weight for bounding box loss.")
    # parser.add_argument("--w_center_loss", type=float, default=0.001, help="Weight for center loss.")
    # parser.add_argument("--w_overlap_loss", type=float, default=0.01, help="Weight for overlap loss.")
    # parser.add_argument("--w_balance_loss", type=float, default=0.01, help="Weight for balance loss.")
    # parser.add_argument("--w_containment_loss", type=float, default=0.01, help="Weight for containment loss")

    parser.add_argument("--mask-mode", type=str, default="mask", help="Model name")

    args = parser.parse_args()

    config = Config(
        epochs=args.epochs,
        lr=args.lr,
        global_batch_size=args.global_batch_size,
        pts_per_shape=args.pts_per_shape,
        npc=args.npc,
        exp_dir=args.exp_dir,
        exp_name=args.exp_name,
        random_seed=args.random_seed,
        pf_dim=args.pf_dim,
        zf_dim=args.zf_dim,
        tf_dim=args.tf_dim,
        ef_dim=args.ef_dim,
        gf_dim=args.gf_dim,
        planef_dim=args.plane_fdim,
        n_planes=args.nplanes,
        n_levels=args.nlevels,
        n_parts=args.nparts,
        model_type=args.model_type,
        mask_mode=args.mask_mode
    )

    return config


@dataclass
class Config:
    epochs: int = 150
    lr: float = 0.0001
    global_batch_size: int = 64
    pts_per_shape: int = 4096
    npc: int = 2048
    exp_dir: str = "./exp"
    exp_name: str = "example"
    random_seed: int = 42

    pf_dim: int = 3
    zf_dim: int = 512
    tf_dim: int = 4
    ef_dim: int = 256
    gf_dim: int = 256

    planef_dim: int = 8
    n_planes: int = 32
    n_levels: int = 3
    n_parts: List[int] = field(default_factory=lambda: [8, 16, 36])

    model_type: str = "hit-volume"
    mask_mode: str = "mask" 