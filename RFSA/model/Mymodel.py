# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : Mymodel.py
# Time       ：2022/11/17 18:10
# Author     ：Dannis
# version    ：python 3.8
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import common
from utils import weights_init_normal

from model.cbam import ChannelAttention, SpatialAttention


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),  # 上上下左右均填充1
                      nn.Conv2d(in_features, in_features, 3),
                      # nn.GroupNorm(8, in_features),
                      # LayerNorm2d(in_features),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      # nn.GroupNorm(8, in_features),
                      # LayerNorm2d(in_features),
                      nn.InstanceNorm2d(in_features),
                      # CALayer(in_features, 16),
                      ]

        self.conv_block = nn.Sequential(*conv_block)
        self.ca = ChannelAttention(in_features)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x + self.conv_block(x)
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x


class Img2DCT(nn.Module):
    def __init__(self, N=8, in_channal=3):
        super(Img2DCT, self).__init__()

        self.N = N  # default is 8 for JPEG
        self.fre_len = N * N
        self.in_channal = in_channal
        self.out_channal = N * N * in_channal

        # rgb to ycbcr
        self.Ycbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        trans_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])
        trans_matrix = torch.from_numpy(trans_matrix).float().unsqueeze(2).unsqueeze(3)
        self.Ycbcr.weight.data = trans_matrix
        self.Ycbcr.weight.requires_grad = False

        # ycbcr to rgb
        self.reYcbcr = nn.Conv2d(3, 3, 1, 1, bias=False)
        re_matrix = np.linalg.pinv(np.array([[0.299, 0.587, 0.114],
                                             [-0.169, -0.331, 0.5],
                                             [0.5, -0.419, -0.081]]))
        re_matrix = torch.from_numpy(re_matrix).float().unsqueeze(2).unsqueeze(3)
        self.reYcbcr.weight.data = re_matrix
        self.reYcbcr.weight.requires_grad = False

        # 3 H W -> 3*N*N  H/N  W/N
        self.dct_conv = nn.Conv2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)

        # 64 *1 * 8 * 8, from low frequency to high fre
        self.weight = torch.from_numpy(self.rearrange(N=N)).float().unsqueeze(1)
        # self.dct_conv = nn.Conv2d(1, self.out_channal, N, N, bias=False)
        self.dct_conv.weight.data = torch.cat([self.weight] * self.in_channal, dim=0)  # 64 1 8 8
        self.dct_conv.weight.requires_grad = False
        # self.norm.weight.requires_grad = False

        self.idct_conv = nn.ConvTranspose2d(self.in_channal, self.out_channal, N, N, bias=False, groups=self.in_channal)
        self.idct_conv.weight.data = torch.cat([self.weight] * self.in_channal, dim=0)
        self.idct_conv.weight.requires_grad = False

        # self.a =
        # self.conv_norn = nn.Conv2d(self.out_channal, self.out_channal, bias=False)
        # self.conv_norn.weight.requires_grad = False

    def norm(self, x, eps=1e-5):
        # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        b, c, _, _ = x.shape
        data_max, data_min = torch.zeros(b, c, requires_grad=False).to(device), \
                             torch.zeros(b, c, requires_grad=False).to(device)

        for i in range(b):
            for j in range(c):
                channel = x[i][j]
                max_channel, min_channel = torch.max(channel), torch.min(channel)
                data_max[i][j], data_min[i][j] = max_channel, min_channel

        x = x - data_min[..., None, None]
        x = x / ((data_max - data_min)[..., None, None] + eps)

        return data_max, data_min, x

    def renorm(self, x, data_max, data_min, eps=1e-5):
        b, c, _, _ = x.shape
        x = x * ((data_max - data_min)[..., None, None] + eps)
        x = x + data_min[..., None, None]
        # x.requires_grad = False
        return x

    def forward(self, x):
        '''
        x:  B C H W, 0-1. RGB
        YCbCr:  b c h w, YCBCR
        DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H
        '''
        ycbcr = self.Ycbcr(x)
        dct = self.dct_conv(ycbcr)
        return dct

    def reverse(self, x):
        dct = self.idct_conv(x)
        rgb = self.reYcbcr(dct)
        return rgb

    def rearrange(self, N=8, zigzag=True):
        dct_weight = np.zeros((N * N, N, N))
        for k in range(N * N):
            u = k // N
            v = k % N
            for i in range(N):
                for j in range(N):
                    tmp1 = self.get_1d(i, u, N=N)
                    tmp2 = self.get_1d(j, v, N=N)
                    tmp = tmp1 * tmp2
                    tmp = tmp * self.get_c(u, N=N) * self.get_c(v, N=N)

                    dct_weight[k, i, j] += tmp
        out_weight = dct_weight
        if zigzag:
            out_weight = self.get_zigzag_order(dct_weight, N=N)  # from low frequency to high frequency
        return out_weight  # (N*N) * N * N

    def get_1d(self, ij, uv, N=8):
        result = math.cos(math.pi * uv * (ij + 0.5) / N)
        return result

    def get_c(self, u, N=8):
        if u == 0:
            return math.sqrt(1 / N)
        else:
            return math.sqrt(2 / N)

    def get_zigzag_order(self, src_weight, N=8):
        array_size = N * N
        # order_index = np.zeros((N, N))
        i = 0
        j = 0
        rearrange_weigth = src_weight.copy()  # (N*N) * N * N
        for k in range(array_size - 1):
            if (i == 0 or i == N - 1) and j % 2 == 0:
                j += 1
            elif (j == 0 or j == N - 1) and i % 2 == 1:
                i += 1
            elif (i + j) % 2 == 1:
                i += 1
                j -= 1
            elif (i + j) % 2 == 0:
                i -= 1
                j += 1
            index = i * N + j
            rearrange_weigth[k + 1, ...] = src_weight[index, ...]
        return rearrange_weigth


class Encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Encoder, self).__init__()

        # Initial convolution block
        model = [  # LayerNorm2d(192),
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            # nn.GroupNorm(8, 64),
            # LayerNorm2d(64),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64

        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, 3, stride=2),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        # out_features = in_features // 2
        # for _ in range(2):
        #     model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #               # nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(1),
                  nn.Conv2d(512, output_nc, 3),  # mean
                  # nn.ReLU()  # max_min
                  nn.Sigmoid()
                  # nn.PReLU(1)
                  ]

        self.model = nn.Sequential(*model)
        # print(self.model)

    def forward(self, x):
        # test_folder = self.model(x)
        # print(test_folder.shape)
        return self.model(x)


class Decoder_msrn(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Decoder_msrn, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 256, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Initial convolution block
        resdual_block = []

        # Residual blocks
        in_features = 256
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))

        conv = common.default_conv

        tail = [
            nn.Conv2d(2560, 64, 1, padding=0, stride=1),
            conv(64, 64, 3),
            common.Upsampler(conv, scale=8, n_feats=64, act=False),
            conv(64, output_nc, 3),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*modules_body)
        self.resdual_block = nn.Sequential(*resdual_block)
        self.tail = nn.Sequential(*tail)
        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        res = x

        RBlock_out = []
        for i in range(9):
            x = self.body[i](x)
            RBlock_out.append(x)
        RBlock_out.append(res)

        res = torch.cat(RBlock_out, 1)
        x = self.tail(res)
        return x


class Decoder_rcan(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Decoder_rcan, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 512, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Initial convolution block
        resdual_block = []

        # Residual blocks
        in_features = 512
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))

        conv = common.default_conv

        tail = [
            nn.Conv2d(512, 64, 1, padding=0, stride=1),
            nn.InstanceNorm2d(64),
            conv(64, 64, 3),
            common.Upsampler(conv, scale=8, n_feats=64, act=False),
            conv(64, 3, 3),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*modules_body)
        self.resdual_block = nn.Sequential(*resdual_block)
        self.tail = nn.Sequential(*tail)
        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        res = x

        # RBlock_out = []
        for i in range(9):
            x = self.body[i](x)
        #     RBlock_out.append(x)
        # RBlock_out.append(res)
        x += res
        # res = torch.cat(RBlock_out, 1)
        x = self.tail(x)
        return x


class Decoder_cbam(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Decoder_cbam, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 512, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Initial convolution block
        resdual_block = []

        # Residual blocks
        in_features = 512
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))

        conv = common.default_conv

        tail = [
            nn.Conv2d(512, 64, 1, padding=0, stride=1),
            nn.InstanceNorm2d(64),
            conv(64, 64, 3),
            common.Upsampler(conv, scale=8, n_feats=64, act=False),
            conv(64, 3, 3),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*modules_body)
        self.resdual_block = nn.Sequential(*resdual_block)

        self.tail = nn.Sequential(*tail)
        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        res = x

        # RBlock_out = []
        for i in range(9):
            x = self.body[i](x)

        #     RBlock_out.append(x)
        # RBlock_out.append(res)

        x += res
        # res = torch.cat(RBlock_out, 1)
        x = self.tail(x)
        return x


if __name__ == '__main__':
    # from torchviz import make_dot
    # from torchsummary import summary

    dct = Img2DCT(N=4)
    # att = AttentionNet()
    # print(att)
    # gen2 = GeneratorImage(input_nc=3, output_nc=3)
    # disc = Discriminator(input_nc=192)
    tmp = torch.arange(2 * 3 * 32 * 32)
    # img = test_folder.resize(2, 3, 32, 32).float()
    # img = torch.randn(2, 3, 32, 32)

    input2 = torch.rand(1, 3, 256, 256)
    outdct = dct(input2)
    rimax, rimin, realImage_inputs = dct.norm(outdct)
    encoded2 = dct.renorm(realImage_inputs, rimax, rimin)
    realImage_outputs = dct.reverse(encoded2)  # [B,3,W,H]
    print(realImage_outputs.shape)
    # attouput = att(input2)
    # print(attouput.shape)

    # out = gen2(input2)

    # a = dct(img)
    # attention_mask = gen2(input2)
    # print(attention_mask.shape)

    encoder = Encoder(input_nc=3, output_nc=192)
    decoder = Decoder_cbam(input_nc=192, output_nc=3)
    output1 = encoder(input2)
    output2 = decoder(output1)
    # print(decoder)
    # print(output1.shape, output2.shape)

    # summary(encoder,(3, 256, 256))
    # summary(decoder,(192, 32, 32))
    # backbone = models.resnet50(pretrained=True)

    # print(backbone)
    # g = make_dot(out)
    # g.render('espnet_model2', view=False)
