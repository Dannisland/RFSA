import os
import time

import numpy
import torch
from PIL import Image
import torchvision.transforms as transforms


from datasets_dct.dataset_imagenet_dct import ImageFolderDCT
import datasets_dct.cvtransforms as cvtransforms
from datasets_dct import train_y_mean_upscaled, train_y_std, train_cb_mean, train_cb_std, \
    train_cr_mean, train_cr_std
from datasets_dct import train_y_mean_upscaled, train_y_std_upscaled, train_cb_mean_upscaled, \
    train_cb_std_upscaled, \
    train_cr_mean_upscaled, train_cr_std_upscaled
from datasets_dct import train_dct_subset_mean, train_dct_subset_std
from datasets_dct import train_upscaled_static_mean, train_upscaled_static_std


def valloader_upscaled_static(args, model='mobilenet'):
    # valdir = os.path.join(args.dataroot, 'train')
    valdir = args.dataroot

    if model == 'mobilenet':
        input_size1 = 1024
        input_size2 = 896
    elif model == 'resnet':
        input_size1 = 512
        input_size2 = 448
    else:
        raise NotImplementedError

    transform2 = transforms.Compose([
        transforms.Resize([args.size, args.size], Image.BICUBIC),
        # transforms.RandomCrop(opt.size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if int(args.subset) == 0 or int(args.subset) == 192:
        transform = cvtransforms.Compose([
            cvtransforms.Resize(input_size1),
            cvtransforms.CenterCrop(input_size2),
            cvtransforms.Upscale(upscale_factor=2),
            cvtransforms.TransformUpscaledDCT(),
            cvtransforms.ToTensorDCT(),
            cvtransforms.Aggregate(),
            cvtransforms.NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
            )
        ])
    else:
        transform = cvtransforms.Compose([
            cvtransforms.Resize(input_size1),
            cvtransforms.CenterCrop(input_size2),
            cvtransforms.Upscale(upscale_factor=2),
            cvtransforms.TransformUpscaledDCT(),
            cvtransforms.ToTensorDCT(),
            cvtransforms.SubsetDCT(channels=args.subset),
            cvtransforms.Aggregate(),
            cvtransforms.NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
                channels=args.subset
            )
        ])

    # val_loader = torch.utils.data.DataLoader(ImageFolderDCT(valdir, transform, target_transform=transform2),
    #                                          batch_size=2, shuffle=False,
    #                                          num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(ImageFolderDCT(valdir, transform, target_transform=transform2), pin_memory=True,
                            batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    return val_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Datasets
    # parser.add_argument('-d', '--data', default='/opt/data/xiaobin/data', type=str)
    parser.add_argument('-d', '--dataroot', default='/opt/data/mingjin/pycharm/Data/ABDH', type=str)
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--subset', default='192', type=str, help='subset of y, cb, cr')
    parser.add_argument('--gpu-id', default='2', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # Device options

    args = parser.parse_args()
    val_loader = valloader_upscaled_static(args, model='resnet')

    for i, batch in enumerate(val_loader):

        a = batch["img"]
        b = batch["sec"]
        c = batch["img_dct_block"]

        print(a,b,c)