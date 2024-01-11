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
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from loguru import logger
from torch import nn
from tqdm import tqdm, tqdm_gui
from tqdm.auto import tqdm as auto_tqdm
from torchinfo import summary

# 本地应用程序库导入
# import datasets
# import transformers
import timm
from core.datasets import *
from core.model import encoder, decoder

# from core.optimzier import *
# from saver import Saver, resume
# from core.dataset import MSRSData, FusionData, initialize_dataloaders
from utils import *


# model
class Encoder_dual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_dual, self).__init__()
        self.vis_encoder = encoder(in_channels, out_channels)
        self.inf_encoder = encoder(in_channels, out_channels)
        self.decoder = decoder(out_channels, in_channels)

    def forward(self, vis_img, inf_img, only_encoder=True):
        feat_vis = self.vis_encoder(vis_img)
        feat_inf = self.inf_encoder(inf_img)
        feat = torch.cat([feat_vis, feat_inf], dim=1)
        # feat = feat_vis + feat_inf
        if not only_encoder:
            feat = self.decoder(feat)
        return [feat_vis, feat_inf, feat]


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
    # cfg = load_cfg_from_cfg_file(args.config)
    try:
        cfg = load_config(args.config)
    except ValueError as e:
        raise e("please specify the config yaml file path with --config")

    return cfg


# load resume of Encoder_dual
configs = get_parser()
ckpt = torch.load("CLF_Net.pth")

# ckpt = torch.load(configs["model"]["encoder_pth"])
encoder_dual = Encoder_dual(
    configs["model"]["in_channels"], configs["model"]["output_channels"]
)

encoder_dual.load_state_dict(ckpt, strict=False)
summary(encoder_dual)
