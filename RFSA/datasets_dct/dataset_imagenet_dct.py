# Optimized for DCT
# Upsampling in the compressed domain
import glob
import os
import sys
import random
from datasets_dct.vision import VisionDataset
from PIL import Image
import cv2
import os.path
import numpy as np
import torch
from turbojpeg import TurboJPEG
from datasets_dct import train_y_mean_resized, train_y_std_resized, train_cb_mean_resized, train_cb_std_resized, \
    train_cr_mean_resized, train_cr_std_resized
from jpeg2dct.numpy import loads


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    d = dir
    # for target in sorted(class_to_idx.keys()):
    #     d = os.path.join(dir, target)
    if not os.path.isdir(d):
        raise ValueError("Dir Error")
    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if is_valid_file(path):
                # item = (path, class_to_idx[target])
                # images.append(item)
                images.append(path)

    return images


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def opencv_loader(path, colorSpace='YCrCb'):
    image = cv2.imread(str(path))
    # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets_dct/cvtransforms/test/raw.jpg', image)
    if colorSpace == "YCrCb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # cv2.imwrite('/mnt/ssd/kai.x/work/code/iftc/datasets_dct/cvtransforms/test/ycbcr.jpg', image)
    elif colorSpace == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def default_loader(path, backend='opencv', colorSpace='YCrCb'):
    from torchvision import get_image_backend
    if backend == 'opencv':
        return opencv_loader(path, colorSpace=colorSpace)
    elif get_image_backend() == 'accimage' and backend == 'acc':
        return accimage_loader(path)
    elif backend == 'pil':
        return pil_loader(path)
    else:
        raise NotImplementedError


def adjust_size(y_size, cbcr_size):
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size


class DatasetFolderDCT(VisionDataset):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None,
                 backend='opencv', mode='train', aggregate=False, unaligned=False):
        super(DatasetFolderDCT, self).__init__(root)

        self.files_img = sorted(glob.glob(os.path.join(root, '%s/img' % mode) + '/*.jpg'))
        self.files_sec = sorted(glob.glob(os.path.join(root, '%s/sec' % mode) + '/*.jpg'))

        self.transform = transform
        self.target_transform = target_transform
        self.unaligned = unaligned

        # classes, class_to_idx = self._find_classes(self.root)
        # samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        # samples = make_dataset(self.root, extensions, is_valid_file)
        if len(self.files_img) == 0 or len(self.files_sec) == 0:
            raise (RuntimeError(
                "Found 0 files in subfolders of: " + self.root + "\n""Supported extensions are: " + ",".join(
                    extensions)))

        self.loader = loader
        self.extensions = extensions
        # self.samples = samples
        self.backend = backend
        self.aggregate = aggregate

    def __getitem__(self, index):
        # 原图的RGB形式，转为tensor数据，范围[0-1]
        item_img = self.target_transform(Image.open(self.files_img[index % len(self.files_img)]).convert('RGB'))  # 输出为像素值正则化到[0-1]
        if self.unaligned:
            item_sec = self.target_transform(
                Image.open(self.files_sec[random.randint(0, len(self.files_sec) - 1)]).convert('RGB'))
        else:
            item_sec = self.target_transform(Image.open(self.files_sec[index % len(self.files_sec)]).convert('RGB'))

        # Cover图像的DCT变换操作
        # path = self.samples[index]
        path = self.files_img[index]
        if self.backend == 'opencv':
            sample = self.loader(path, backend='opencv', colorSpace='BGR')
        elif self.backend == 'dct':
            try:
                with open(path, 'rb') as src:
                    buffer = src.read()
                dct_y, dct_cb, dct_cr = loads(buffer)
            except:
                notValid = True
                while notValid:
                    index = random.randint(0, len(self.files_img) - 1)
                    # path, target = self.samples[index]
                    path = self.files_img[index]
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        notValid = True

            if len(dct_y.shape) != 3:
                notValid = True
                while notValid:
                    index = random.randint(0, len(self.files_img) - 1)
                    # path, target = self.samples[index]
                    path = self.files_img[index]
                    with open(path, 'rb') as src:
                        buffer = src.read()
                    try:
                        dct_y, dct_cb, dct_cr = loads(buffer)
                        notValid = False
                    except:
                        print(path)
                        notValid = True

            y_size_h, y_size_w = dct_y.shape[:-1]
            cbcr_size_h, cbcr_size_w = dct_cb.shape[:-1]

            y_size_h, cbcr_size_h = adjust_size(y_size_h, cbcr_size_h)
            y_size_w, cbcr_size_w = adjust_size(y_size_w, cbcr_size_w)
            dct_y = dct_y[:y_size_h, :y_size_w]
            dct_cb = dct_cb[:cbcr_size_h, :cbcr_size_w]
            dct_cr = dct_cr[:cbcr_size_h, :cbcr_size_w]
            sample = [dct_y, dct_cb, dct_cr]

            y_h, y_w, _ = dct_y.shape
            cbcr_h, cbcr_w, _ = dct_cb.shape

        if self.transform is not None:
            dct_y, dct_cb, dct_cr = self.transform(sample)

        if self.backend == 'dct':
            if dct_cb is not None:
                image = torch.cat((dct_y, dct_cb, dct_cr), dim=1)
                # return image, target
                return image
            else:
                # return dct_y, target
                return dct_y
        else:
            if dct_cb is not None:
                # return dct_y, dct_cb, dct_cr, target
                return dct_y, dct_cb, dct_cr
            else:
                # return dct_y, target
                return {'img': item_img, 'sec': item_sec, 'img_dct_block': dct_y}

    def __len__(self):
        return max(len(self.files_img), len(self.files_sec))


class ImageFolderDCT(DatasetFolderDCT):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, backend='opencv', aggregate=False):
        super(ImageFolderDCT, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                             transform=transform,
                                             target_transform=target_transform,
                                             is_valid_file=is_valid_file,
                                             backend=backend,
                                             aggregate=aggregate)
        self.files_img = self.files_img
        self.files_sec = self.files_sec


if __name__ == '__main__':
    dataset = 'imagenet'

    import torch
    import datasets_dct.cvtransforms as transforms
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import minmax_scale

    # jpeg_encoder = TurboJPEG('/home/kai.x/work/local/lib/libturbojpeg.so')
    jpeg_encoder = TurboJPEG('/usr/lib/libturbojpeg.so')
    if dataset == 'imagenet':
        input_normalize = []
        input_normalize_y = transforms.Normalize(mean=train_y_mean_resized,
                                                 std=train_y_std_resized)
        input_normalize_cb = transforms.Normalize(mean=train_cb_mean_resized,
                                                  std=train_cb_std_resized)
        input_normalize_cr = transforms.Normalize(mean=train_cr_mean_resized,
                                                  std=train_cr_std_resized)
        input_normalize.append(input_normalize_y)
        input_normalize.append(input_normalize_cb)
        input_normalize.append(input_normalize_cr)
        val_loader = torch.utils.data.DataLoader(
            # ImageFolderDCT('/mnt/ssd/kai.x/dataset/ILSVRC2012/val', transforms.Compose([
            ImageFolderDCT('/opt/data/xiaobin/data/val', transforms.Compose([
                # transforms.ToYCrCb(),
                # transforms.TransformDCT(),
                transforms.TransformUpscaledDCT(),
                # transforms.UpsampleDCT(T=896, debug=False),
                transforms.DCTCenterCrop(112),
                transforms.ToTensorDCT(),
                transforms.NormalizeDCT(
                    train_y_mean_resized, train_y_std_resized,
                    train_cb_mean_resized, train_cb_std_resized,
                    train_cr_mean_resized, train_cr_std_resized),
            ])),
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)

        # train_dataset = ImageFolderDCT('/mnt/ssd/kai.x/dataset/ILSVRC2012/train', transforms.Compose([
        train_dataset = ImageFolderDCT('/opt/data/xiaobin/data/train', transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ToYCrCb(),
            # transforms.ChromaSubsample(),
            # transforms.UpsampleDCT(size=224, T=896, debug=False),
            transforms.Upscale(upscale_factor=2),
            transforms.TransformUpscaledDCT(),
            transforms.ToTensorDCT(),
            transforms.NormalizeDCT(
                train_y_mean_resized, train_y_std_resized,
                train_cb_mean_resized, train_cb_std_resized,
                train_cr_mean_resized, train_cr_std_resized),
        ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)

    from torchvision.utils import save_image

    dct_y_mean_total, dct_y_std_total = [], []
    # for batch_idx, (dct_y, dct_cb, dct_cr, targets) in enumerate(val_loader):
    for batch_idx, (dct_y, dct_cb, dct_cr, targets) in enumerate(train_loader):
        coef = dct_y.numpy()
        dct_y_mean, dct_y_std = [], []

        for c in coef:
            c = c.reshape((64, -1))
            dct_y_mean.append([np.mean(x) for x in c])
            dct_y_std.append([np.std(x) for x in c])

        dct_y_mean_np = np.asarray(dct_y_mean).mean(axis=0)
        dct_y_std_np = np.asarray(dct_y_std).mean(axis=0)
        dct_y_mean_total.append(dct_y_mean_np)
        dct_y_std_total.append(dct_y_std_np)
        # print('The mean of dct_y is: {}'.format(dct_y_mean_np))
        # print('The std of dct_y is: {}'.format(dct_y_std_np))

    print('The mean of dct_y is: {}'.format(np.asarray(dct_y_mean_total).mean(axis=0)))
    print('The std of dct_y is: {}'.format(np.asarray(dct_y_std_total).mean(axis=0)))
