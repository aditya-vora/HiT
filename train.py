import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import argparse 
import pprint

from config import *
from utils.distributed import init_distributed_mode
from utils.logger import create_logger
from utils import *
# from src.models.hit import HiTModel
from src.models.hit import HiT_models
from src.loss import Loss
from dataset.build import build_dataset


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(config: Config, loss_config: LossConfig): 
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(config)
    
    assert config.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    
    # you get the device from ddp
    device = rank % torch.cuda.device_count()    
    if rank == 0: 
        pprint.pprint(config)
        
    # set device and random seed
    seed = config.random_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    checkpoint_dir = None
    if rank == 0:
        experiment_dir = setup_experiment_directory(base_dir=config.exp_dir, exp_name=config.exp_name)
        checkpoint_dir = make_path_and_dir(folders=[experiment_dir, "checkpoints"])
        logger = create_logger(experiment_dir, filename='log', mode="ddp")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, None, mode="ddp")

    # training args
    logger.info(f"{config}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    model = HiT_models[config.model_type](
        pf_dim=config.pf_dim,
        zf_dim=config.zf_dim,
        tf_dim=config.tf_dim,
        ef_dim=config.ef_dim,
        gf_dim=config.gf_dim,
        n_parts=config.n_parts,
        n_levels=config.n_levels,
        n_planes=config.n_planes,
        planef_dim=config.planef_dim,
        mask_mode=config.mask_mode
    ).to(device=device)

    logger.info(f"Model Encoder Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Model Encoder Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad == True):,}")


    loss = Loss(
        w_im_sample_loss=loss_config.w_im_sample_loss,
        w_equilibrium_loss=loss_config.w_equilibrium_loss,
        w_sample_loss=loss_config.w_sample_loss,
        w_bbx_loss=loss_config.w_bbx_loss,
        w_center_loss=loss_config.w_center_loss,
        w_overlap_loss=loss_config.w_overlap_loss,
        w_balance_loss=loss_config.w_balance_loss,
        w_containment_loss=loss_config.w_containment_loss,
        n_levels=loss_config.n_levels,
        n_parts=loss_config.n_parts,
    ).to(device=device)

    if config.cats == "all":
        config.cats_list = read_text(f"{config.data_dir}/cats.txt")
    else:
        config.cats_list = [config.cats]

    logger.info(f"Training on categories: {config.cats_list}")

    dataset = build_dataset(config=config, mode="train")
    pass

if __name__ == "__main__":
    config, loss_config = parse_args()
    main(config=config, loss_config=loss_config)