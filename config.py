import argparse 
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class LossConfig:
    w_im_sample_loss: float = 1.0
    w_equilibrium_loss: float = 0.001
    w_sample_loss: float = 1.0
    w_bbx_loss: float = 0.01
    w_center_loss: float = 0.001
    w_overlap_loss: float = 0.01
    w_balance_loss: float = 0.01
    w_containment_loss: float = 0.01
    n_levels: int = 3
    n_parts: List[int] = field(default_factory=lambda: [8,16,36])

@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 0.0001
    global_batch_size: int = 64
    pts_per_shape: int = 4096
    npc: int = 2048
    exp_dir: str = "./exp"
    exp_name: str = "example"
    random_seed: int = 42
    ckpt: str = ""


@dataclass
class ModelConfig: 
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
    train_mode: bool = True
    recon_mode: bool = False
    pc_mode: bool = False
    mesh_mode: bool = False
    iou_mode: bool = False
    cd_mode: bool = False

@dataclass 
class DataConfig: 
    data_dir: str = "/mnt/data/ava/hit/data/objaverse"
    dataset_name: str = "objaverse"
    cats: str = "objaverse"
    shape_id: str = "1ab3abb5c090d9b68e940c4e64a94e1e"
    cats_list: List[str] = field(default_factory=list)
    splits: List[str] = field(default_factory=lambda: ["train", "val"])

def parse_args() -> Tuple[DataConfig, ModelConfig, TrainConfig, LossConfig]:
    parser = argparse.ArgumentParser(description="PyTorch HiT Training")
    
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for Adam optimizer [0.0001]")
    parser.add_argument("--global-batch-size", type=int, default=24, help="Batch size")
    parser.add_argument("--pts-per-shape", type=int, default=4096, help="Number of points per shape.")
    parser.add_argument("--npc", type=int, default=2048, help="Number of points in the point cloud.")
    parser.add_argument("--exp-dir", type=str, default="/mnt/data/ava/hit/per_cat_exp", help="Directory for saving experiments")
    parser.add_argument("--exp-name", type=str, default="sample", help="Directory for saving experiments")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ckpt", type=str, default="", help="Directory for saving checkpoints")

    parser.add_argument("--data-dir", type=str, default="/localhome/ava40/Desktop/HiT/data_src/shapenet/shapenetv2.1", help="Root directory of dataset")
    parser.add_argument("--dataset-name", type=str, default="shapenet", help="Name of the dataset")
    parser.add_argument("--cats", type=str, default="all", help="Category of the object")
    parser.add_argument("--shape-id", type=str, default="1ab3abb5c090d9b68e940c4e64a94e1e", help="Shape ID of the object")
    parser.add_argument("--splits", type=str, nargs='+', default=["train", "val"], help="Dataset splits to use")

    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--recon", action="store_true", help="Output reconstructed shape with segmentation")
    parser.add_argument("--pointcloud", action="store_true", help="Output point cloud with segmentation")
    parser.add_argument("--mesh", action="store_true", help="Output mesh with segmentation")
    parser.add_argument("--iou", action="store_true", help="Output IOU for test shapes")
    parser.add_argument("--cd", action="store_true", help="Output chamfer distance for test shapes")

    # parser.add_argument("--log_data", type=int, default=100, help="Directory for saving log data")
    # parser.add_argument("--log_model", type=int, default=1000, help="Directory for saving log model")
    # parser.add_argument("--check_val_every_n_epoch", type=int, default=10, help="Check validation every n epochs")
    parser.add_argument("--nparts", type=int, nargs='+', default=[8,16,36], help="Number of parts")
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

    # loss params 
    parser.add_argument("--w-im-sample-loss", type=float, default=1.0, help="Weight for MLP occ decoder loss.")
    parser.add_argument("--w-sample-loss", type=float, default=1.0, help="Weight for convex sample loss.")
    parser.add_argument("--w-equilibrium-loss", type=float, default=0.001, help="Weight for equilibrium loss.")
    parser.add_argument("--w-bbx-loss", type=float, default=0.01, help="Weight for bounding box loss.")
    parser.add_argument("--w-center-loss", type=float, default=0.001, help="Weight for center loss.")
    parser.add_argument("--w-overlap-loss", type=float, default=0.01, help="Weight for overlap loss.")
    parser.add_argument("--w-balance-loss", type=float, default=0.01, help="Weight for balance loss.")
    parser.add_argument("--w-containment-loss", type=float, default=0.01, help="Weight for containment loss")

    parser.add_argument("--mask-mode", type=str, default="mask", help="Model name")

    args = parser.parse_args()

    dataconfig = DataConfig(
        data_dir=args.data_dir,
        dataset_name=args.dataset_name,
        cats=args.cats,
        splits=args.splits,
        shape_id=args.shape_id
    )

    modelconfig = ModelConfig(
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
        mask_mode=args.mask_mode,
        train_mode=args.train,
        recon_mode=args.recon,
        pc_mode=args.pointcloud,
        mesh_mode=args.mesh,
        iou_mode=args.iou,
        cd_mode=args.cd,
    )

    trainconfig = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        global_batch_size=args.global_batch_size,
        pts_per_shape=args.pts_per_shape,
        npc=args.npc,
        exp_dir=args.exp_dir,
        exp_name=args.exp_name,
        random_seed=args.random_seed,
        ckpt=args.ckpt
    )

    loss_config = LossConfig(
        w_im_sample_loss=args.w_im_sample_loss,
        w_equilibrium_loss=args.w_equilibrium_loss,
        w_sample_loss=args.w_sample_loss,
        w_bbx_loss=args.w_bbx_loss,
        w_center_loss=args.w_center_loss,
        w_overlap_loss=args.w_overlap_loss,
        w_balance_loss=args.w_balance_loss,
        w_containment_loss=args.w_containment_loss,
        n_levels=args.nlevels,
        n_parts=args.nparts,
    )

    return (dataconfig, modelconfig, trainconfig, loss_config)
