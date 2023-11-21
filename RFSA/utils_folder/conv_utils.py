# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : conv_utils.py
# Time       ：2022/9/7 14:41
# Author     ：Dannis
# version    ：python 3.8 https://github.com/JuZiSYJ/fancy_operations
"""

import cv2
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F

import utils_DCT


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


class MaxMinNormalize(object):
    def __init__(self, tensor):
        # pass
        # '''tensor: B*C*W*H'''
        max_, min_ = utils_DCT.cal_channel_max_min(tensor)

        self.max_ = max_
        self.min_ = min_

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for i in range(tensor.shape[0]):
            for t, m, s in zip(tensor[i], self.max_[i], self.min_[i]):
                t.sub_(s).div_(m - s + 1e-6)
        return tensor

    def reverse(self, tensor):
        for i in range(tensor.shape[0]):
            for t, m, s in zip(tensor[i], self.max_[i], self.min_[i]):
                t.mul_(m - s + 1e-6).add_(s)
        return tensor


def UnMaxMinNormalize(tensor):
    '''tensor: B*C*W*H'''
    max_, min_ = utils_DCT.cal_channel_max_min(tensor)
    out = tensor.clone()
    for i in range(out.shape[0]):
        for t, m, s in zip(out, max_[i], min_[i]):
            t.mul_(m - s + 1e-6).add_(s)
    return out


class DCT(nn.Module):
    def __init__(self, N=8, in_channal=3, *args):
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

    def forward(self, x, max_, min_):
        '''
        x:  B C H W, 0-1. RGB
        YCbCr:  b c h w, YCBCR
        DCT: B C*64 H//8 W//8 ,   Y_L..Y_H  Cb_L...Cb_H   Cr_l...Cr_H
        '''
        ycbcr = self.Ycbcr(x)
        dct = self.dct_conv(ycbcr)
        out = dct.clone()
        for i in range(out.shape[0]):
            for t, m, s in zip(out, max_, min_):
                t.sub_(s).div_(m - s + 1e-6)
        return out

    def reverse(self, x):
        out = UnMaxMinNormalize(x)
        dct = self.idct_conv(out)
        rgb = self.reYcbcr(dct)
        return rgb

    def rearrange(self, N=8, zigzag=False):
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


def test_dct():
    '''
    new version of DCT
    :return:
    '''
    N = 56  # The block size of DCT ,default is 8
    Test_DCT = DCT(N=N)
    # img = np.array(Image.open('/opt/data/xiaobin/ABDH_UDH/ABDH/realimg.png'))
    img = cv2.imread('/opt/data/xiaobin/ABDH_UDH/ABDH/realimg.png')
    img = cv2.resize(img, [64, 64]) / 255.0

    img_tensor = torch.tensor(img).float().unsqueeze(0).permute(0, 3, 1, 2)  # BCHW

    # from RGB to DCT domain,  [R_L1, R_L2, .... R_H, G_L1, G_L2, G_H,..], channels is 64 *3
    img_192 = img_tensor.reshape(1,192,8,-1)
    max_min = MaxMinNormalize(img_192)
    out = max_min(img_192)
    dct_ = Test_DCT(out)

    # redct = Test_DCT.reverse(dct_)  # == img_tensor
    # ttt = np.array(redct.detach())

    print(dct_.shape)
    print('N:{}   DCT shape:{}'.format(N, dct_.shape))


if __name__ == '__main__':
    test_dct()
