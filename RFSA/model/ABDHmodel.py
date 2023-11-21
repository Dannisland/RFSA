# !/usr/bin/env python
# -*-coding:utf-8 -*-
# import os

from model.cbam import ChannelAttention, SpatialAttention
#
# # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import common
from test_folder.layer import MultiSpectralAttentionLayer
from utils import weights_init_normal


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm on channels for 2d images.
    Args:
        num_channels (int): The number of channels of the input tensor.
        eps (float): a value added to the denominator for numerical stability.
            Defaults to 1e-5.
        elementwise_affine (bool): a boolean value that when set to ``True``,
            this module has learnable per-element affine parameters initialized
            to ones (for weights) and zeros (for biases). Defaults to True.
    """

    def __init__(self, num_channels: int, **kwargs) -> None:
        super().__init__(num_channels, **kwargs)
        self.num_channels = self.normalized_shape[0]

    def forward(self, x):
        assert x.dim() == 4, 'LayerNorm2d only supports inputs with shape ' \
                             f'(N, C, H, W), but got tensor with shape {x.shape}'
        return F.layer_norm(
            x.permute(0, 2, 3, 1), self.normalized_shape, self.weight,
            self.bias, self.eps).permute(0, 3, 1, 2)


class DCT(nn.Module):
    def __init__(self, N=8, in_channal=3):
        super(DCT, self).__init__()

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
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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


class AttentionNet_jpeg(nn.Module):
    def __init__(self):
        super(AttentionNet_jpeg, self).__init__()
        """
        resnet50的结构：conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
                       1/2              1/2           1/2    1/2    1/2  
        取到layer2止  
        不下采样
        """
        backbone = models.resnet50(pretrained=True)
        model = list(backbone.children())[:6]

        self.feature_extractor = nn.Sequential(*model)
        # print(self.feature_extractor)
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(512, 256, 3),
                                   nn.InstanceNorm2d(256),
                                   nn.ReLU(True))

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 64, 3),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   nn.Sigmoid())
        self.conv1.apply(weights_init_normal)
        self.conv2.apply(weights_init_normal)
        self.conv3.apply(weights_init_normal)

        # conv1 = self.feature_extractor[0]
        # conv1.stride = (1, 1)

        layer2_block1 = self.feature_extractor[5][0]
        layer2_block1.conv2.stride = (1, 1)
        layer2_block1.downsample[0].stride = (1, 1)

        self.fc = nn.Sequential(
            nn.Linear(256, 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(16, 256, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # 得到第一个feature_map1
        x = self.feature_extractor(x)
        x = self.conv1(x)
        n, c, h, w = x.shape

        fc_input = torch.sum(x, dim=[2, 3])
        fc_output = self.fc(fc_input).view(n, c, 1, 1)
        channel_attention = fc_output

        # x = self.conv1(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv2(x)
        x = self.conv3(x)
        return channel_attention, x


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        """
        resnet50的结构：conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
                       1/2              1/2           1/2    1/2    1/2  
        取到layer2止  
        不下采样
        """
        backbone = models.resnet50(pretrained=True)
        model = list(backbone.children())[:6]
        # in_features = 512
        # out_features = in_features // 2
        # for _ in range(2):
        #     model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2
        # model += [nn.ReflectionPad2d(3),
        #           nn.Conv2d(128, 1, 7),
        #           nn.Sigmoid()]
        # in_features = 512
        # out_features = in_features // 2
        # for _ in range(5):
        #     model += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #               nn.ReflectionPad2d(1),
        #               nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=0),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2
        self.feature_extractor = nn.Sequential(*model)
        # print(self.feature_extractor)
        # self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
        #                            nn.Conv2d(512, 256, 3),
        #                            nn.InstanceNorm2d(256),
        #                            nn.ReLU(True))
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(512, 256, 3),
                                   nn.InstanceNorm2d(256),
                                   nn.ReLU(True),
                                   nn.Conv2d(256, 256, 1),
                                   nn.Sigmoid()
                                   )
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(256, 64, 3),
                                   nn.InstanceNorm2d(64),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 3, 1),
                                   nn.Sigmoid())
        self.conv1.apply(weights_init_normal)
        self.conv2.apply(weights_init_normal)
        self.conv3.apply(weights_init_normal)

        # conv1 = self.feature_extractor[0]
        # conv1.stride = (1, 1)

        layer2_block1 = self.feature_extractor[5][0]
        layer2_block1.conv2.stride = (1, 1)
        layer2_block1.downsample[0].stride = (1, 1)

        # layer3_block1 = self.feature_extractor[6][0]
        # layer3_block1.conv2.stride = (1, 1)
        # layer3_block1.downsample[0].stride = (1, 1)
        #
        # layer4_block1 = self.feature_extractor[6][0]
        # layer4_block1.conv2.stride = (1, 1)
        # layer4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):  # 得到第一个feature_map1
        x = self.feature_extractor(x)
        x = self.conv1(x)

        # x = self.conv1(F.interpolate(x, scale_factor=2, mode='nearest'))
        # x = self.conv2(F.interpolate(x, scale_factor=2, mode='nearest'))
        # x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(in_features)

        self.att = MultiSpectralAttentionLayer(in_features, 14, 14, reduction=16, freq_sel_method='top16')

        # self.conv_jnd = nn.Sequential(
        #     nn.Conv2d(3, 64, 7, 1, 3),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.InstanceNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )

        # self.ca_jnd = ChannelAttention(in_features)

        # conv_block = [nn.ReflectionPad2d(1),  # 上上下左右均填充1
        #               nn.Conv2d(in_features, in_features, 3),
        #               # LayerNorm2d(in_features),
        #               nn.InstanceNorm2d(in_features),
        #               nn.ReLU(inplace=True),
        #               nn.ReflectionPad2d(1),
        #               nn.Conv2d(in_features, in_features, 3),
        #               # LayerNorm2d(in_features),
        #               nn.InstanceNorm2d(in_features),
        #               ]

        self.ca = ChannelAttention(in_features)
        self.sa = SpatialAttention()

    def forward(self, x):
        # old
        res = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        # out = self.ca(out) * out
        # 改成：
        out = self.att(out)

        out = self.sa(out) * out

        out = out + res
        out = self.relu(out)


        # res = x
        # out = self.conv1(x)
        # out = self.in1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.in2(out)
        #
        # if jnd is not None:
        #     j_out = self.ca_jnd(jnd)
        #     out = j_out * out
        #
        # # out = self.att(out)
        #
        # out = self.ca(out) * out
        # out = self.sa(out) * out
        # # out = channel_a * out
        # # out = spatial_a * out
        #
        # out = out + res
        # out = self.relu(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, stride=1, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)  # torch.sigmoid(
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)  # [b,1]


class Encoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dct_kernel=4, n_residual_blocks=9):
        super(Encoder, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Downsampling
        downsample = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(int(math.log(dct_kernel, 2))):
            downsample += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, 3, stride=2),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.downsample = nn.Sequential(*downsample)

        # Residual blocks
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, output_nc, 3),  # mean
            # nn.ReLU()  # max_min
            nn.Sigmoid()
        )

    def forward(self, x):
        """old forward"""
        x = self.conv1(x)
        x = self.downsample(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x

        # y = self.conv1_jpeg(jpeg)
        #
        # x = self.conv1(x)
        # x = self.downsample(x)
        # res = x
        #
        # x = x + y
        # for i in range(self.n_residual_blocks - 1):
        #     x = self.body[i](x) + y
        # x = self.body[8](x)
        #
        # x = x + res
        # x = self.tail(x)
        # return x


class Encoder2(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dct_kernel=4, n_residual_blocks=9):
        super(Encoder2, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        # self.conv_jnd = nn.Sequential(
        #     nn.Conv2d(3, 64, 7, 1, 3),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 256, 3, 1, 1),
        #     nn.InstanceNorm2d(256),
        #     nn.ReLU(inplace=True)
        # )

        # Initial convolution block
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Downsampling
        downsample = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(int(math.log(dct_kernel, 2))):
            downsample += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_features, out_features, 3, stride=2),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.downsample = nn.Sequential(*downsample)

        # Residual blocks
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))
        self.body = nn.Sequential(*modules_body)

        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, output_nc, 3),  # mean
            # nn.ReLU()  # max_min
            nn.Sigmoid()
        )

    def forward(self, x, jnd, channel_a=1, spatial_a=1):
        """old forward"""
        x = self.conv1(x)
        x = self.downsample(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x
        # if jnd != None:
        #     jnd = self.conv_jnd(jnd)
        # # j_out = self.ca(j_out)
        #
        # x = self.conv1(x)
        # x = self.downsample(x)
        # res = x
        #
        # for i in range(self.n_residual_blocks):
        #     x = self.body[i](x, jnd, channel_a, spatial_a)
        #
        #
        # x = x + res
        # x = self.tail(x)
        # return x


class Decoder(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9, dct_kernel=8):
        super(Decoder, self).__init__()

        self.upconv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.upconv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upconv3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 256, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Initial convolution block
        model = []
        # model = [  # LayerNorm2d(192),
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(input_nc, 512, 7),
        #     # nn.GroupNorm(8, 64),
        #     # LayerNorm2d(64),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True)]

        in_features = 256

        # Upsampling
        # out_features = in_features // 2
        # for _ in range(3):
        #     model += [nn.ConvTranspose2d(in_features, out_features, 4, stride=2, padding=1),
        #               # nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
        #               nn.InstanceNorm2d(out_features),
        #               nn.ReLU(inplace=True)]
        #     in_features = out_features
        #     out_features = in_features // 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Output layer
        tail = [nn.ReflectionPad2d(1),
                nn.Conv2d(64, output_nc, 3),  # mean
                # nn.ReLU()  # max_min
                nn.Sigmoid()
                # nn.PReLU(1)
                ]

        self.model = nn.Sequential(*model)
        self.tail = nn.Sequential(*tail)

        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        x = self.model(x)
        x = self.upconv1(F.interpolate(x, scale_factor=2, mode='bilinear'))
        x = self.upconv2(F.interpolate(x, scale_factor=2, mode='bilinear'))
        x = self.upconv3(F.interpolate(x, scale_factor=2, mode='bilinear'))
        x = self.tail(x)
        return x


class Decoder_cbam(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, dct_kernel=8, n_residual_blocks=9):
        super(Decoder_cbam, self).__init__()

        self.n_residual_blocks = n_residual_blocks
        tmp_channel = 128

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, tmp_channel, 7),
            nn.InstanceNorm2d(tmp_channel),
            nn.ReLU(inplace=True)
        )

        # Residual blocks
        in_features = tmp_channel
        modules_body = nn.ModuleList()
        for i in range(n_residual_blocks):
            modules_body.append(
                ResidualBlock(in_features))

        conv = common.default_conv

        tail = [
            nn.Conv2d(tmp_channel, 64, 1, padding=0, stride=1),
            conv(64, 64, 3),
            common.Upsampler(conv, scale=dct_kernel, n_feats=64, act=False),
            conv(64, output_nc, 3),
            nn.Sigmoid()
        ]

        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*tail)
        # print(self.model)

    def forward(self, x):
        x = self.conv1(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x


if __name__ == '__main__':
    # from torchviz import make_dot
    # from torchsummary import summary
    device = torch.device("cuda:3")

    dct = DCT(N=4)

    att = AttentionNet()
    att_jpeg = AttentionNet_jpeg()

    # print(att)
    # gen2 = GeneratorImage(input_nc=3, output_nc=3)
    disc = Discriminator(input_nc=192)
    tmp = torch.arange(2 * 3 * 32 * 32)
    # img = test_folder.resize(2, 3, 32, 32).float()
    # img = torch.randn(2, 3, 32, 32)

    input2 = torch.rand(1, 3, 256, 256)
    jpeg = torch.rand(1, 3, 256, 256)
    jnd = torch.rand(1, 3, 8, 8)
    dct_ = dct(input2)
    attouput = att(input2)
    ca, sa = att_jpeg(input2)

    # print(attouput.shape)

    # out = gen2(input2)

    # a = dct(img)
    # attention_mask = gen2(input2)
    # print(attention_mask.shape)

    encoder = Encoder2(input_nc=3, output_nc=48, dct_kernel=4)
    decoder = Decoder_cbam(input_nc=48, output_nc=3, dct_kernel=4)
    output1 = encoder(input2, None, ca, sa)
    output2 = decoder(output1)
    print(encoder)
    print(output1.shape, output2.shape)

    # summary(encoder,(3, 256, 256))
    # summary(decoder,(192, 32, 32))
    # backbone = models.resnet50(pretrained=True)

    # print(backbone)
    # g = make_dot(out)
    # g.render('espnet_model2', view=False)
