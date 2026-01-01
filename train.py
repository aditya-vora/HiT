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
def main(dataconfig: DataConfig, modelconfig: ModelConfig, trainconfig: TrainConfig, lossconfig: LossConfig): 
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(trainconfig)
    
    assert trainconfig.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    
    # you get the device from ddp
    device = rank % torch.cuda.device_count()    
    if rank == 0: 
        pprint.pprint(trainconfig)
        pprint.pprint(modelconfig)
        pprint.pprint(lossconfig)
        pprint.pprint(dataconfig)
        
    # set device and random seed
    seed = trainconfig.random_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    checkpoint_dir = None
    if rank == 0:
        experiment_dir = setup_experiment_directory(base_dir=trainconfig.exp_dir, exp_name=trainconfig.exp_name)
        checkpoint_dir = make_path_and_dir(folders=[experiment_dir, "checkpoints"])
        logger = create_logger(experiment_dir, filename='log', mode="ddp")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, None, mode="ddp")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    model = HiT_models[modelconfig.model_type](
        pf_dim=modelconfig.pf_dim,
        zf_dim=modelconfig.zf_dim,
        tf_dim=modelconfig.tf_dim,
        ef_dim=modelconfig.ef_dim,
        gf_dim=modelconfig.gf_dim,
        n_parts=modelconfig.n_parts,
        n_levels=modelconfig.n_levels,
        n_planes=modelconfig.n_planes,
        planef_dim=modelconfig.planef_dim,
        mask_mode=modelconfig.mask_mode
    ).to(device=device)

    logger.info(f"Model Encoder Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Model Encoder Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad == True):,}")


    loss = Loss(
        w_im_sample_loss=lossconfig.w_im_sample_loss,
        w_equilibrium_loss=lossconfig.w_equilibrium_loss,
        w_sample_loss=lossconfig.w_sample_loss,
        w_bbx_loss=lossconfig.w_bbx_loss,
        w_center_loss=lossconfig.w_center_loss,
        w_overlap_loss=lossconfig.w_overlap_loss,
        w_balance_loss=lossconfig.w_balance_loss,
        w_containment_loss=lossconfig.w_containment_loss,
        n_levels=lossconfig.n_levels,
        n_parts=lossconfig.n_parts,
    ).to(device=device)

    if dataconfig.cats == "all":
        dataconfig.cats_list = read_text(f"{dataconfig.data_dir}/cats.txt")
    else:
        dataconfig.cats_list = [dataconfig.cats]

    logger.info(f"Training on categories: {dataconfig.cats_list}")
    dataset = build_dataset(config=config, mode="train")
    pass

if __name__ == "__main__":
    dataconfig, modelconfig, trainconfig, lossconfig = parse_args()
    main(dataconfig=dataconfig, modelconfig=modelconfig, trainconfig=trainconfig, lossconfig=lossconfig)