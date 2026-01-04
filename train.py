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
from src.models.hit import HiT_models
from src.loss import Loss
from src.dataset.build import build_dataset


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(data_config: DataConfig, model_config: ModelConfig, train_config: TrainConfig, loss_config: LossConfig): 
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(train_config)
    
    assert train_config.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    
    # you get the device from ddp
    device = rank % torch.cuda.device_count()    
    if rank == 0: 
        pprint.pprint(train_config)
        pprint.pprint(model_config)
        pprint.pprint(loss_config)
        pprint.pprint(data_config)
        
    # set device and random seed
    seed = train_config.random_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    checkpoint_dir = None
    if rank == 0:
        experiment_dir = setup_experiment_directory(base_dir=train_config.exp_dir, exp_name=train_config.exp_name)
        checkpoint_dir = make_path_and_dir(folders=[experiment_dir, "checkpoints"])
        logger = create_logger(experiment_dir, filename='log', mode="ddp")
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, None, mode="ddp")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

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

    if data_config.cats == "all":
        data_config.cats_list = read_text(f"{data_config.data_dir}/cats.txt")
    else:
        data_config.cats_list = [data_config.cats]

    logger.info(f"Training on categories: {data_config.cats_list}")
    dataset = build_dataset(
        config=data_config, 
        train_mode=model_config.train_mode,
        recon_mode=model_config.recon_mode,
        pc_mode=model_config.pc_mode,
        mesh_mode=model_config.mesh_mode,
        iou_mode=model_config.iou_mode,
        cd_mode=model_config.cd_mode
    )
    # pass

if __name__ == "__main__":
    dataconfig, modelconfig, trainconfig, lossconfig = parse_args()
    main(data_config=dataconfig, model_config=modelconfig, train_config=trainconfig, loss_config=lossconfig)