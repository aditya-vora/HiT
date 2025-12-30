import logging
import torch.distributed as dist

def create_logger(logging_dir, filename, mode: str="ddp") -> logging.Logger:
    """
    Create a logger that writes to a log file and stdout.
    Args:
        logging_dir (str): Directory to save the log file.
        filename (str): Name of the log file (without extension).
    Returns:
        logging.Logger: Configured logger instance.

    :NOTE In distributed settings, only the process with rank 0 will log messages;
    In case of single device processes, create logger without checking dist.rank().
    """
    if mode == "ddp":
        if dist.get_rank() == 0:  # real logger
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{filename}.txt")]
            )
            logger = logging.getLogger(__name__)
        else:  # dummy logger (does nothing)
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.NullHandler())
    else: # single process logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/{filename}.txt")]
        )
        logger = logging.getLogger(__name__)
    return logger