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


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(config: Config): 
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
    )

    logger.info(f"Model Encoder Parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
    logger.info(f"Model Encoder Trainable Parameters: {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad == True):,}")


    pass

if __name__ == "__main__":
    args = parse_args()
    main(config=args)