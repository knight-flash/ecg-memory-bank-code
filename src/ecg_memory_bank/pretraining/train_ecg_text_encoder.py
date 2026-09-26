"""Train the paper's ECG–text representation from external HEEDB inputs.

Run with ``torchrun --nproc_per_node=N``. Weights and data stay outside the
source archive.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

import utils_builder
from dataset import HEEDBDataset
from utils_trainer import trainer_wBert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--medcpt-model", type=Path, required=True)
    parser.add_argument("--wandb", action="store_true", help="Enable optional run logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.train_csv, args.val_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    from ecg_memory_bank.protocol import external_output_dir

    output_dir = external_output_dir(args.output_dir)
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["network"]["ecg_model"] != "ecgfounder" or config["network"]["text_model"] != "medcpt":
        raise ValueError("The paper training entry requires ecgfounder and medcpt")
    config["trainer"]["checkpoint"] = str(output_dir) + os.sep
    if not args.medcpt_model.is_dir():
        raise FileNotFoundError(args.medcpt_model)
    config["network"]["medcpt_model"] = str(args.medcpt_model)

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch this training entry with torchrun")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(42)
    np.random.seed(0)
    random.seed(0)

    if rank == 0:
        import wandb

        wandb.init(
            project="ECG-Memory-Bank",
            name=config["wandb_name"],
            config=config,
            mode="online" if args.wandb else "disabled",
        )

    train_dataset = HEEDBDataset(indexed_csv_path=str(args.train_csv))
    val_dataset = HEEDBDataset(indexed_csv_path=str(args.val_csv))
    model = utils_builder.ECGCLIP(config["network"])
    frozen_layers = config["network"].get("free_layers")
    if frozen_layers is not None:
        for layer in list(model.lm_model.encoder.layer)[: int(frozen_layers)]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
    model = DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"]["params"], betas=(0.9, 0.999))
    trainer = trainer_wBert(
        model=model,
        optimizer=optimizer,
        device=rank,
        model_name=config["wandb_name"],
        **config["trainer"],
    )
    try:
        trainer.train_w_TextEmb(train_dataset, val_dataset, config["zeroshot"])
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
