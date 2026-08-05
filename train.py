import time
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

    loss_func = Loss(
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

    # only optimize the vq model parameters. Currently a simple Adam optimizer is used.
    optimizer = torch.optim.Adam( # type: ignore
        params=model.parameters(), 
        lr=train_config.lr, 
        betas=(0.9, 0.95)
    )

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

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=train_config.random_seed
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Read from: {data_config.data_dir}, Dataset contains {len(dataset):,} images.")

    # load the model from a checkpoint is training breaks.
    if train_config.ckpt != "":
        checkpoint = torch.load(train_config.ckpt, map_location="cpu")
        
        model.load_state_dict(checkpoint["model"])
        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(train_config.ckpt.split('/')[-1].split('.')[0])
        
        start_epoch = int(train_steps / int(len(dataset) / train_config.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / train_config.global_batch_size))
        optimizer.load_state_dict(checkpoint["optimizer"])

        del checkpoint
        logger.info(f"Resume training from checkpoint: {train_config.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
           
    model = DDP(model.to(device), device_ids=[getattr(train_config, "gpu")])
    model.train()
    
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {train_config.epochs} epochs...")
    for epoch in range(start_epoch, train_config.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for i, data in enumerate(dataloader):           
            # move data to device
            querypts = data['querypts'].to(device, non_blocking=True)
            occ = data['occgt'].to(device, non_blocking=True)
            points = data['pc'].to(device, non_blocking=True)
            fileids = data['fileid']
            
            optimizer.zero_grad()

            n_query_pts = querypts.shape[1] 
            data = model(points, querypts, fileids)
            
            loss = loss_func(data, occ, querypts, points, n_query_pts, iter_num=train_steps)
            loss['total_loss'].backward()

            if train_config.max_grad_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    train_config.max_grad_norm
                )
                
            optimizer.step()

            # # Log loss values:
            running_loss += loss['total_loss'].item() 

            log_steps += 1
            train_steps += 1

            if train_steps % train_config.log_every == 0:

                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % train_config.ckpt_every == 0 and train_steps > 0:
                # as ddp loads model on each gpu, we need to save only from rank 0
                # we only save the vq model weights.
                if rank == 0:
                    assert checkpoint_dir is not None, "checkpoint_dir should not be None for rank 0"
                    model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "train_args": train_config,
                        "data_args": data_config,
                        "model_args": model_config,
                        "loss_args": loss_config
                    }
                    ckpt_path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
                    torch.save(checkpoint, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")                    
                dist.barrier()


if __name__ == "__main__":
    dataconfig, modelconfig, trainconfig, lossconfig, _ = parse_args()
    main(data_config=dataconfig, model_config=modelconfig, train_config=trainconfig, loss_config=lossconfig)