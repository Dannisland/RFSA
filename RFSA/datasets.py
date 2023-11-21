# -*- coding:utf-8 -*-
# 小何一定行
import glob
import random
import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from utils import same_seeds
from utils_folder import utils_DCT
from utils_folder.jnd_dct import cal_jnd_dct


def jpeg_image_residuals(input_img):
    # 调整通道维度
    img_np = np.array(input_img).transpose(1, 2, 0)

    # JPEG变换
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encimg = cv2.imencode('.jpg', img_np * 255, encode_param)
    decimg_255 = cv2.imdecode(encimg, cv2.IMREAD_COLOR)  # jpeg变换的图像
    decimg = decimg_255.astype(np.float32) / 255

    # 疑问？ 输入的是【-1，1】会不会影响结果？
    redidual = img_np - decimg

    return redidual


def DCT_CoverImg_domain(path):
    transform = transforms.Compose([transforms.Resize((128, 128), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    ])
    img = transform(Image.open(path).convert('RGB'))

    img_dct = np.zeros(img.shape)
    img_np = img.numpy()
    for i in range(3):
        img_channel = img_np[i, :, :]
        img_channel_dct = cv2.dct(np.array(img_channel, np.float32))
        img_dct[i, :, :] = img_channel_dct
        # test_folder = cv2.idct(img_channel_dct)
    img_dct_tenosr = torch.from_numpy(img_dct)

    # -- recv = H * W * C -- #
    # re_cv = img_dct.numpy()
    # for i in range(3):
    #     re_channel = re_cv[i, :, :]
    #     re_idct = cv2.idct(re_channel.astype(np.float32))
    #     recv[:, :, i] = re_idct
    # recv = np.around(recv).astype(np.uint8)
    # print("1")

    # ---- DCT 残差(原始图像和JPEG压缩后的图像在dct域下生成的残差)
    # for i in range(3):
    #     # 原图像求DCT
    #     img_channel = img_np[:, :, i]
    #     img_channel_dct = cv2.dct(np.array(img_channel, np.float32))
    #
    #     # 变换后图像求DCT
    #     decimg_channel = decimg[:, :, i]
    #     decimg_channel_dct = cv2.dct(np.array(decimg_channel, np.float32))
    #
    #     # 计算残差并IDCT返回
    #     diff_channel = img_channel_dct - decimg_channel_dct
    #     diff_channel_idct = cv2.idct(diff_channel)
    #     # diff_channel_idct = diff_channel_idct.astype(np.uint8)
    #     # img_target[:, :, i] = diff_channel_idct

    return img_dct_tenosr


class ImageDataset(Dataset):
    def __init__(self, path, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_img = sorted(glob.glob(os.path.join(path, '%s/img' % mode) + '/*.jpg'))
        self.files_sec = sorted(glob.glob(os.path.join(path, '%s/sec' % mode) + '/*.jpg'))
        # self.files_img = sorted(glob.glob(os.path.join(path, 'train2017') + '/*.jpg'))
        # self.files_sec = sorted(glob.glob(os.path.join(path, 'val2017') + '/*.jpg'))

        # self.files_img = sorted(glob.glob(path + '/*.png'))
        # self.files_sec = sorted(glob.glob(path + '/*.png'))
        # print(self.test_folder)

    def __getitem__(self, index):
        item_img = self.transform(
            Image.open(self.files_img[index % len(self.files_img)]).convert('RGB'))  # 输出为像素值正则化到[0-1]
        # print("-----path----",  self.files_img[index % len(self.files_img)]) # '/opt/data/mingjin/pycharm/Data/ABDH/train/img/000000379820.jpg'

        path_img = self.files_img[index % len(self.files_img)]

        # JPEG残差
        # jpeg_residual = jpeg_image_residuals(item_img)  # 残差值
        # jpeg_residual = jpeg_residual.transpose(2, 0, 1)
        # jpeg_residual = torch.from_numpy(jpeg_residual)

        # jnd_dct
        # jnd = cal_jnd_dct(item_img).astype(np.float32)
        #
        # jnd = torch.from_numpy(jnd)

        if self.unaligned:
            item_sec = self.transform(
                Image.open(self.files_sec[random.randint(0, len(self.files_sec) - 1)]).convert('RGB'))
        else:
            item_sec = self.transform(Image.open(self.files_sec[index % len(self.files_sec)]).convert('RGB'))

        # return {'img': item_img, 'sec': item_sec, 'img_dct_block': item_img_dct, 'img_diff': img_diff}
        return {'img': item_img, 'sec': item_sec} # , 'jpeg_residual': jpeg_residual, 'jnd': jnd}

    def __len__(self):
        return max(len(self.files_img), len(self.files_sec))


if __name__ == '__main__':
    transforms_ = [transforms.Resize([128, 128], Image.BICUBIC),
                   # transforms.RandomCrop(256),
                   # transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 空域使用
                   ]
    path = '/opt/data/mingjin/pycharm/Data/ABDH'
    # path = '/workshop/Datasets/COCO' # '/opt/data/mingjin/pycharm/Data/ABDH'
    dataloader = DataLoader(ImageDataset(path, transforms_=transforms_, unaligned=True),
                            batch_size=5, shuffle=False, num_workers=0)
    print(dataloader)
    # inputs = next(iter(dataloader))

    for i, inputs in enumerate(dataloader):
        print(inputs['img'].shape)
        print(inputs['sec'].shape)
        print(inputs['jpeg_residual'].shape)
        # print(inputs['high_spectral_block'].shape)
        # print(inputs['low_spectral_block'].shape)
