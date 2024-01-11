""" From CLF-Net Encoder && IRFS(2023)"""
import sys

sys.path.append("../")
import torch
import torch.nn as nn
import numpy as np

from .module import *


class encoder(nn.Module):
    """docstring for dense"""

    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=5, stride=1, padding=2)
        # self.bn = nn.BatchNorm2d(self.out_channels)
        # self.lrelu = nn.LeakyReLU(inplace=True)
        self.res_block0 = Residual_Block(
            self.in_channels, self.out_channels, identity=True
        )
        self.res_block1 = Residual_Block(
            self.out_channels, self.out_channels, identity=False
        )
        self.res_block2 = Residual_Block(
            self.out_channels, self.out_channels * 2, identity=True
        )
        self.res_block3 = Residual_Block(
            self.out_channels * 2, self.out_channels * 4, identity=True
        )

    def forward(self, x):
        # feat = self.conv1(x)
        # feat = self.bn(feat)
        # feat = self.lrelu(feat)
        feat = self.res_block0(x)
        feat = self.res_block1(feat)
        feat = self.res_block2(feat)
        feat = self.res_block3(feat)
        return feat


# IRFS
# ---------- Dual Attention Unit ----------
class DualAttEncoder(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualAttEncoder, self).__init__()
        modules_body = [
            conv(1, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias),
        ]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res


# ---------- Dual Attention Edge Unit ----------
class DualEdgeAttEncoder(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualEdgeAttEncoder, self).__init__()
        modules_body = [
            conv(1, n_feat, kernel_size, bias=bias),
            act,
            conv(n_feat, n_feat, kernel_size, bias=bias),
        ]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)
        ## Sobel Edge
        self.Edge = Sobelxy(n_feat)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res_att = torch.cat([sa_branch, ca_branch], dim=1)
        res_att = self.conv1x1(res_att)
        edge = self.Edge(res)
        out = res_att + edge
        return out


# test
if __name__ == "__main__":
    pass
