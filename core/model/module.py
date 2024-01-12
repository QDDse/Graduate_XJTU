""" Some Important blocks"""
import math

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from utils import to_2tuple


# from DMA
class ResDown(nn.Module):
    # residual downsample module
    def __init__(
        self,
        args,
        in_channel=32,
        out_channel=32,
        kernel=(3, 3),
        down_stride=(2, 2),
        stride=(1, 1),
        padding=(1, 1),
        bias=True,
    ):
        super(ResDown, self).__init__()
        self.channel = args.feature_num
        self.res = nn.Sequential(
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=down_stride,
                padding=padding,
                bias=bias,
            ),
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )
        self.down_conv = nn.Sequential(
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=down_stride,
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
        )

    def forward(self, x):
        res = self.res(x)
        x_down = self.down_conv(x)
        out = res + x_down
        return out


class ResUp(nn.Module):
    # residual upsample module
    def __init__(
        self,
        in_channel=32,
        out_channel=32,
        kernel=(3, 3),
        up_stride=(2, 2),
        stride=(1, 1),
        padding=(1, 1),
        bias=True,
    ):
        super(ResUp, self).__init__()
        self.channel = args.feature_num
        self.res = nn.Sequential(
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
            nn.ConvTranspose2d(
                self.channel,
                self.channel,
                kernel,
                stride=up_stride,
                padding=padding,
                output_padding=(1, 1),
                bias=bias,
            ),
            nn.Conv2d(
                self.channel,
                out_channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.ConvTranspose2d(
                self.channel,
                out_channel,
                kernel,
                stride=up_stride,
                padding=padding,
                output_padding=(1, 1),
                bias=bias,
            ),
            nn.PReLU(),
        )

    def forward(self, x):
        res = self.res(x)
        x_up = self.up_conv(x)
        out = res + x_up
        return out


class ChannelAtt(nn.Module):
    # channel attention module
    def __init__(self, args, stride, reduction=8, bias=True):
        super(ChannelAtt, self).__init__()
        self.channel = args.feature_num
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale
        self.channel_down = nn.Sequential(
            nn.Conv2d(
                self.channel, self.channel // reduction, (1, 1), padding=0, bias=bias
            ),
            nn.PReLU(),
        )
        # feature upscale --> channel weight
        self.channel_up1 = nn.Sequential(
            nn.Conv2d(
                self.channel // reduction, self.channel, (1, 1), padding=0, bias=bias
            ),
            nn.Sigmoid(),
        )
        self.channel_up2 = nn.Sequential(
            nn.Conv2d(
                self.channel // reduction, self.channel, (1, 1), padding=0, bias=bias
            ),
            nn.Sigmoid(),
        )
        # different resolution to same
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel,
                self.channel,
                (3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=(1, 1),
                bias=bias,
            ),
            nn.PReLU(),
        )

    def forward(self, x, y):
        if x.shape[2] < y.shape[2]:
            x = self.up(x)
        fusion = torch.add(x, y)
        fusion = self.channel_down(self.avg_pool(fusion))
        out_x = self.channel_up1(fusion)
        out_y = self.channel_up2(fusion)
        return [out_x, out_y]


class SpatialAtt(nn.Module):
    # spatial attention module
    def __init__(self, args, stride, kernel=(3, 3), padding=(1, 1), bias=True):
        super(SpatialAtt, self).__init__()
        self.channel = args.feature_num
        self.trans_conv = nn.Sequential(
            # nn.Conv2d(self.channel, self.channel, kernel, stride=(1, 1), padding=padding, bias=bias),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ConvTranspose2d(
                self.channel,
                self.channel,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=(1, 1),
                bias=bias,
            ),
            nn.PReLU(),
        )
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(
                self.channel * 2,
                self.channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=0,
                bias=bias,
            ),
            # nn.BatchNorm2d(in_channel, eps=1e-5, momentum=0.01, affine=True),
            nn.PReLU(),
        )
        self.down = nn.Sequential(
            nn.Conv2d(
                self.channel,
                self.channel,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                bias=bias,
            ),
            nn.PReLU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel,
                self.channel,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=(1, 1),
                bias=bias,
            ),
            nn.Sigmoid(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.channel,
                self.channel,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=padding,
                output_padding=(1, 1),
                bias=bias,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        if x.shape[2] < y.shape[2]:
            x = self.trans_conv(x)
        fusion = torch.cat([x, y], dim=1)
        fusion = self.down(self.conv_fusion(fusion))
        up_x = self.up1(fusion)
        up_y = self.up2(fusion)
        return [up_x, up_y]


# from CLF-Net
class Residual_Block:
    def __init__(self, i_channel, o_channel, identity=None, end=False):
        super(Residual_Block, self).__init__()

        self.in_channels = i_channel
        self.out_channels = o_channel
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        # self.conv3 = nn.Conv2d(in_channels=self.in_channels,
        # 					   out_channels=self.out_channels ,
        # 					   kernel_size=3,
        # 					   stride=1,
        # 					   padding=1,
        # 					   bias=False)
        # self.bn3 = nn.BatchNorm2d(self.out_channels)

        self.identity_block = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.identity = identity
        self.end = end
        self.tanh = nn.Tanh()
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 将单元的输入直接与单元输出加在一起
        if self.identity:
            residual = self.identity_block(x)
        out += residual
        if self.end:
            out = self.tanh(out)
        else:
            out = self.lrelu(out)
        return out


class decoder(nn.Module):
    """docstring for dense"""

    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_block1 = Residual_Block(
            self.in_channels * 8, self.in_channels * 4, identity=True
        )
        self.res_block2 = Residual_Block(
            self.in_channels * 4, self.in_channels * 2, identity=True
        )
        self.res_block3 = Residual_Block(
            self.in_channels * 2, self.in_channels, identity=True
        )
        self.res_block4 = Residual_Block(
            self.in_channels, self.out_channels, identity=True, end=True
        )

    def forward(self, x):
        feat = self.res_block1(x)
        feat = self.res_block2(feat)
        feat = self.res_block3(feat)
        feat = self.res_block4(feat)


# from IRFS
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


# ---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=False,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


##########################################################################
# ------ Spatial Attention --------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


##########################################################################
# ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##---------- Sobel Edge Unit ----------
class Sobelxy(nn.Module):
    def __init__(
        self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1
    ):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        self.convx = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]}).",
        )
        assert (
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]}).",
        )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
