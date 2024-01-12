""" Training Phase 1 -- patch-align """
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
import timm
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from loguru import logger
from tqdm import tqdm, tqdm_gui
from tqdm.auto import tqdm as auto_tqdm
from torchinfo import summary
import torch.utils.data as data
import wandb

# 本地应用程序库导入
# import datasets
# import transformers
from core.datasets import *
from core.model import encoder, decoder, DualAttEncoder, DualEdgeAttEncoder
from core.datasets import TrainKaist
from core.loss import PA_loss
from core.optim import Optimizer

# from core.optimzier import *
# from saver import Saver, resume
# from core.dataset import MSRSData, FusionData, initialize_dataloaders
from utils import *


# config
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
        "--config",
        default="/root/autodl-tmp/Graduate_XJTU/config/fusion.yaml",
        type=str,
        help="config file",
    )

    args = parser.parse_args()
    # assert args.config is not None
    # configs = load_cfg_from_cfg_file(args.config)
    try:
        configs = load_config(args.config)
    except ValueError as e:
        raise e("please specify the config yaml file path with --config")

    return configs


def train():
    # load resume of Encoder_dual
    configs = get_parser()
    wandb.init(
        project="Dual_encoder_patchAlign",
        name="dual_encoder_patch_align",
        entity="qddse",
        config=configs,
    )
    ckpt = torch.load(configs["model"]["encoder_pth"])

    # model
    dual_att_encoder = DualEdgeAttEncoder(configs, configs["model"]["n_feat"])

    dual_att_encoder.load_state_dict(ckpt, strict=False)
    dual_att_encoder.cuda()
    # summary(dual_att_encoder)

    # dataset
    dataset_train = TrainKaist(
        configs["Data_Acquisition"]["kaist_path"],
        patch=configs["Data_Acquisition"]["patch"],
    )

    # dataloader
    train_configs = configs["train_config"]

    train_loader = data.DataLoader(
        dataset_train,
        batch_size=train_configs["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    #  random seed
    seed = np.random.randint(2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if train_configs["deterministic"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device(
        "cuda:{}".format(configs["gpu"]) if torch.cuda.is_available() else "cpu"
    )

    # 先加载dataloader 计算每个epoch的的迭代步数 然后计算总的迭代步数
    ep_iter = len(train_loader)
    max_iter = train_configs["num_epoch"] * ep_iter

    # print(f"==== TEST ep_iter:{ep_iter} \n train_list_len:{len(next(iter(train_loader)))} \ntrain_img.shape:{next(iter(train_loader))[0].shape}")
    print("Training iter: {}".format(max_iter))

    # momentum = train_configs["momentum"]
    weight_decay = train_configs["weight_decay"]
    lr_start = train_configs["lr_start"]
    max_iter = 150000
    power = train_configs["power"]
    warmup_steps = train_configs["warmup_steps"]
    warmup_start_lr = train_configs["warmup_start_lr"]

    optimizer = Optimizer(
        model=dual_att_encoder,  ## TODO: 完成dual_att_encoder backbone 的搭建
        lr0=lr_start,
        # momentum=momentum,  # Adam no need
        wd=weight_decay,
        warmup_steps=warmup_steps,
        warmup_start_lr=warmup_start_lr,
        max_iter=max_iter,
        power=power,
    )
    # 训练循环
    best_loss = 1e5
    for epoch in range(train_configs["num_epoch"]):
        epoch_loss = 0.0
        # 使用 tqdm 进行训练循环的进度条显示
        with tqdm(train_loader, unit="iter") as tepoch:
            for ir, vis in tepoch:
                tepoch.set_description(f"Epoch {epoch}")

                # 将数据移动到 GPU
                ir = ir.to(device)
                vis = vis.to(device)

                # 清空梯度
                optimizer.zero_grad()

                # 前向传播
                ir_features = dual_att_encoder(ir)
                vis_features = dual_att_encoder(vis)

                # 计算损失函数
                loss = PA_loss(ir_features, vis_features)
                epoch_loss += loss.item()

                # 反向传播
                loss.backward()
                optimizer.step()

                # 更新进度条
                tepoch.set_postfix(loss=loss.item())
                wandb.log({f"batch_loss_ep{epoch}": loss.item()})

        # 计算平均loss
        epoch_loss /= len(train_loader)

        wandb.log({"epoch_loss": epoch_loss})

        # is best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                dual_att_encoder.state_dict(), "./checkpoint/patch_align/best_model.pth"
            )

        # 每两个epoch保存一次模型
        if (epoch + 1) % 2 == 0:
            torch.save(
                dual_att_encoder.state_dict(),
                f"./checkpoint/patch_align/model_epoch_{epoch+1}.pth",
            )
            wandb.save(f"model_epoch_{epoch+1}.pth")


train()
