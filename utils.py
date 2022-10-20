import logging, colorama
import torch

def init_logger(logger):
    logger.setLevel(logging.INFO)
    log_format = (
        colorama.Fore.MAGENTA
        + "[%(asctime)s %(name)s %(levelname)s] "
        + colorama.Fore.WHITE
        + "%(message)s"
    )
    local_rank = torch.distributed.get_rank()
    if local_rank == 0:
        logging.basicConfig(
            format=log_format, level=logging.INFO, datefmt="%I:%M:%S"
        )
    else:
        logging.basicConfig(
            format=log_format, level=logging.CRITICAL, datefmt="%I:%M:%S"
        )
    return logger, local_rank


def init_logger_nonddp(logger):
    logger.setLevel(logging.INFO)
    log_format = (
        colorama.Fore.MAGENTA
        + "[%(asctime)s %(name)s %(levelname)s] "
        + colorama.Fore.WHITE
        + "%(message)s"
    )
    logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%I:%M:%S")

    return logger