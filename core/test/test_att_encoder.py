import sys

sys.path.append("../")

import torch

import torch.nn as nn
import pytest
from model import DualAttEncoder, DualEdgeAttEncoder


# 初始化
def test_init():
    n_feat = 64
    encoder = DualEdgeAttEncoder(n_feat)
    assert encoder is not None, "Init failed!"


# 测试 forward
def test_forward():
    n_feat = 64
    kernel_size = 3
    reduction = 8
    bias = False
    act = nn.PReLU()

    encoder = DualEdgeAttEncoder(n_feat, kernel_size, reduction, bias, act)
    dummy_input = torch.rand(1, 1, 128, 128)  # 假设输入是单通道图像
    output = encoder(dummy_input)
    assert output is not None, "Forward pass failed!"
    assert output.shape == torch.Size(
        [1, n_feat, 128, 128]
    ), f"Output shape mismatch: {output.shape}"


# load checkpoint的测试
def test_load_checkpoint():
    n_feat = 64
    encoder = DualEdgeAttEncoder(n_feat)
    checkpoint_path = "../../checkpoint/fusion_model.pth"  # 替换为你的checkpoint路径

    # 假设checkpoint是一个包含模型权重的字典，键为'state_dict'
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(checkpoint, strict=False)
    assert encoder is not None, "Loading checkpoint failed!"
