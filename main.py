# 标准库导入
import yaml
import argparse
import datetime
import logging
import math
import os
import random
import warnings
from collections import OrderedDict
from time import time
from typing import Tuple, Union

# 第三方库导入
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from loguru import logger
from torch import nn
from tqdm import tqdm, tqdm_gui
from tqdm.auto import tqdm as auto_tqdm

# 本地应用程序库导入
import datasets
import diffusers
import timm
import transformers
from datasets import *
from core.model import UNet2DConditionModel
from core.optimzier import *
from saver import Saver, resume
from core.dataset import MSRSData, FusionData, initialize_dataloaders
from utils import *

# 以及不清楚`TimestepEmbedding, Timesteps`应该从哪里导入。
# from diffusers.models.embeddings import TimestepEmbedding, Timesteps# option parser


# -----------------------------------------------------------------------------
# Functions for parsing args
# ----------------------------------------------------------------------------
def load_config(config_path):
    assert os.path.isfile(config_path) and config_path.endswith(
        ".yaml"
    ), "{} is not a yaml file".format(config_path)

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def get_parser():
    parser = argparse.ArgumentParser(description="DIFFusion Pytorch Vis and Inf fusion")
    parser.add_argument(
        "--config", default="path to xxx.yaml", type=str, help="config file"
    )

    args = parser.parse_args()
    # assert args.config is not None
    # cfg = load_cfg_from_cfg_file(args.config)
    try:
        cfg = load_config(args.config)
    except ValueError as e:
        raise e("please specify the config yaml file path with --config")

    return cfg


@logger.catch
def main():
    cfg = get_parser()
    print(f"==== Config ===== \n {cfg}")
    #  random seed
    seed = np.random.randint(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # define model, optimzier, scheduler
    device = torch.device(
        "cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu"
    )
    fusion_model = FusionModel(cfg).to(device)  ## TODO: Fusion backbone 搭建

    # initialize dataset
    # TODO: 完成Dataset的搭建 （直接参考CLF-Data，MetaFusion）
    train_loader, test_loader = initialize_dataloaders(cfg)

    ## 先加载dataloader 计算每个epoch的的迭代步数 然后计算总的迭代步数
    ep_iter = len(train_loader)
    max_iter = cfg.num_epoch * ep_iter

    ## test
    # print(f"==== TEST ep_iter:{ep_iter} \n train_list_len:{len(next(iter(train_loader)))} \ntrain_img.shape:{next(iter(train_loader))[0].shape}")
    print("Training iter: {}".format(max_iter))

    momentum = cfg.momentum
    weight_decay = cfg.weight_decay
    lr_start = cfg.lr_start
    # max_iter = 150000
    power = cfg.power
    warmup_steps = cfg.warmup_steps
    warmup_start_lr = cfg.warmup_start_lr

    optimizer = Optimizer(
        model=fusion_model,  ## TODO: 完成fusion_model backbone 的搭建
        lr0=lr_start,
        momentum=momentum,
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
    )

    # Test: 测试这里的resume 可用性
    if cfg.resume:
        fusion_model, optimizer.optim, ep, total_it = resume(
            fusion_model, optimizer.optim, cfg.resume, device
        )
        optimizer = Optimizer(
            model=fusion_model,
            lr0=lr_start,
            momentum=momentum,
            wd=weight_decay,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            max_iter=max_iter,
            power=power,
            it=total_it,
        )
        lr = optimizer.get_lr()
        print("lr:{}".format(lr))
    else:
        ep = -1
        total_it = 0
    ep += 1

    # training
    # TODO: 搭建Training Loop （参考https://wandb.ai/wandb_fc/tips/reports/How-To-Write-Efficient-Training-Loops-in-PyTorch--VmlldzoyMjg4OTk5#show-me-the-code）
    train()


if __name__ == "__main__":
    main()
