import logging, colorama
import torch
import torch.nn as nn
from transformers.trainer import Trainer, get_parameter_names

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

class ChIDTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            if self.args.frozen_backbone:
                optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n.find("fused_cls") != -1 or n.find("prefix") != -1],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * 10
                },
                # {
                #     "params": [p for n, p in self.model.named_parameters() if n.find("prefix") != -1],
                #     "weight_decay": self.args.weight_decay,
                #     "lr": self.args.learning_rate * 10
                # },
            ]
            else:
                decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
                decay_parameters = [name for name in decay_parameters if "bias" not in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if n.find("fused_cls") != -1],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate * 10
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n.find("fused_cls") == -1 and n in decay_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if n.find("fused_cls") == -1 and n not in decay_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer