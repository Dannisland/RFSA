""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cbam import ChannelAttention, SpatialAttention
from test_folder.layer import MultiSpectralAttentionLayer


class DoubleCv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if norm_layer != None:
            norm = norm_layer(mid_channels)
        else:
            norm = nn.BatchNorm2d(mid_channels)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm,
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleCv(in_channels, out_channels, norm_layer=norm_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm_layer=None):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleCv(in_channels, out_channels, in_channels // 2, norm_layer=norm_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleCv(in_channels, out_channels, norm_layer=norm_layer)

        # self.conv2 = DoubleCv(in_channels * 2, in_channels, norm_layer=norm_layer)

    def forward(self, x1, x2, y):
        if y != None:
            x1 = x1 + y
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        # if y != None:
        #     x1 = torch.cat([x1, y], dim=1)
        #     x1 = self.conv2(x1)
        # x1 = self.up(x1)
        #
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        #
        # x = torch.cat([x2, x1], dim=1)




        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class AddOutConv(nn.Module):
#     def __init__(self, input_nc, output_nc, nhf=64, norm_layer=None):
#         super(AddOutConv, self).__init__()
#         self.conv1 = nn.Conv2d(input_nc, nhf, 3, 1, 1)
#         self.conv2 = nn.Conv2d(nhf, nhf * 2, 3, 1, 1)
#         self.conv3 = nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1)
#         self.conv4 = nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1)
#         self.conv5 = nn.Conv2d(nhf * 2, nhf, 3, 1, 1)
#         self.conv6 = nn.Conv2d(nhf, output_nc, 3, 1, 1)
#         self.output = nn.Sigmoid()
#         self.relu = nn.ReLU(True)
#
#         self.norm_layer = norm_layer
#         if norm_layer != None:
#             self.norm1 = norm_layer(nhf)
#             self.norm2 = norm_layer(nhf * 2)
#             self.norm3 = norm_layer(nhf * 4)
#             self.norm4 = norm_layer(nhf * 2)
#             self.norm5 = norm_layer(nhf)
#
#     def forward(self, x, y):
#         x = x+y
#
#         if self.norm_layer != None:
#             x = self.relu(self.norm1(self.conv1(x)))
#             x = self.relu(self.norm2(self.conv2(x)))
#             x = self.relu(self.norm3(self.conv3(x)))
#             x = self.relu(self.norm4(self.conv4(x)))
#             x = self.relu(self.norm5(self.conv5(x)))
#             x = self.output(self.conv6(x))
#         else:
#             x = self.relu(self.conv1(x))
#             x = self.relu(self.conv2(x))
#             x = self.relu(self.conv3(x))
#             x = self.relu(self.conv4(x))
#             x = self.relu(self.conv5(x))
#             x = self.output(self.conv6(x))
#
#         return x

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(in_features)

        self.ca = ChannelAttention(in_features)
        self.att = MultiSpectralAttentionLayer(in_features, 14, 14, reduction=16, freq_sel_method='top16')

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

        return out


class AddOutCv(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(AddOutCv, self).__init__()

        self.n_residual_blocks = n_residual_blocks

        # Initial convolution block
        # self.conv1 = nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     nn.Conv2d(input_nc, 64, 7),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )

        # Downsampling
        downsample = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
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

    def forward(self, x, y):
        """old forward"""
        if y != None:
            x = x + y
        # x = self.conv1(x)
        x = self.downsample(x)
        res = x
        for i in range(self.n_residual_blocks):
            x = self.body[i](x)
        x = x + res
        x = self.tail(x)
        return x
