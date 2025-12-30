import os
import torch
import subprocess
from loguru import logger
from config import Config

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args: Config):
    """Setup distributed training mode

    Args:
        args (Config): Input training configuration dataclass. Contains distributed training parameters.
    """
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        setattr(args, "rank", int(os.environ["RANK"]))
        setattr(args, "world_size", int(os.environ['WORLD_SIZE']))
        setattr(args, "gpu", int(os.environ['LOCAL_RANK']))
        setattr(args, "dist_url", 'env://')
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        setattr(args, "rank", proc_id)
        setattr(args, "world_size", ntasks)
        setattr(args, "gpu", proc_id % num_gpus)
        setattr(args, "dist_url", 'env://')
    else:
        logger.info('Not using distributed mode')
        setattr(args, "distributed", False)
        return

    setattr(args, "distributed", True)

    torch.cuda.set_device(getattr(args, "gpu"))
    setattr(args, "dist_backend", 'nccl')

    logger.info('| distributed init (rank {}): {}'.format(
        getattr(args, "rank"),
        getattr(args, "dist_url")
    ), flush=True)
    
    torch.distributed.init_process_group(
        backend=getattr(args, "dist_backend"),
        init_method=getattr(args, "dist_url"),
        world_size=getattr(args, "world_size"),
        rank=getattr(args, "rank")
    )
    torch.distributed.barrier()
    setup_for_distributed(getattr(args, "rank") == 0)