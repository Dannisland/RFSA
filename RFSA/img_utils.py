import os
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
import torch


def TO_IDCT(img):
    img_np = img.cpu().detach().numpy()  # 3,128,128
    img_each_result = np.zeros((3, 128, 128))  # 3,128,128

    for j in range(3):
        img_each_channel = img_np[j, :, :]
        img_each_channel = cv2.idct(img_each_channel)

        img_each_result[j, :, :] = img_each_channel

    return img_each_result


def ycbcr2rgb(ycbcr_image):
    """
    convert ycbcr into rgb, output = RGB, shape = [w, h, 3]
    """
    if type(ycbcr_image) == torch.Tensor:
        ycbcr_image = np.array(ycbcr_image).astype(np.float32)

    if len(ycbcr_image.shape) != 3 or ycbcr_image.shape[2] != 3:
        if ycbcr_image.shape[0] == 3:
            ycbcr_image = ycbcr_image.transpose([1, 2, 0])
        else:
            raise ValueError("input image is not a rgb image")

    ycbcr_image = ycbcr_image.astype(np.float32)
    ycbcr_image = ycbcr_norm(ycbcr_image)
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    transform_matrix_inv = np.linalg.inv(transform_matrix)
    shift_matrix = np.array([16, 128, 128])
    rgb_image = np.zeros(shape=ycbcr_image.shape)
    w, h, _ = ycbcr_image.shape
    for i in range(w):
        for j in range(h):
            rgb_image[i, j, :] = np.dot(transform_matrix_inv, ycbcr_image[i, j, :]) - np.dot(transform_matrix_inv,
                                                                                             shift_matrix)
    # 得到的RGB数值规范化到[0-255]
    # rgb_image_ = cv2.normalize(rgb_image, None, 0, 255, cv2.NORM_MINMAX)
    # rgb_image_ = np.around(rgb_image_)
    # rgb_image_ = rgb_image_.astype(np.uint8)

    rgb_image = np.around(rgb_image)
    rgb_image = rgb_image.astype(np.uint8)
    return rgb_image


def ycbcr_norm(ycbcr_image):
    """
    Normalized YCBCR values, output = YCBCR, shape = [w, h, 3]
    """
    if type(ycbcr_image) == torch.Tensor:
        ycbcr_image = np.array(ycbcr_image).astype(np.float32)

    if len(ycbcr_image.shape) != 3 or ycbcr_image.shape[0] != 3:
        if ycbcr_image.shape[2] == 3:
            ycbcr_image = ycbcr_image.transpose([2, 0, 1])
        else:
            raise ValueError("The shape of the input ycbcr image is wrong")

    ycbcr_image = ycbcr_image.astype(np.float32)
    y_channel, cb_channel, cr_channel = ycbcr_image[0], ycbcr_image[1], ycbcr_image[2]
    y_channel = np.clip(y_channel, 16, 235)
    cb_channel = np.clip(cb_channel, 16, 240)
    cr_channel = np.clip(cr_channel, 16, 240)
    ycbcr_image_ = np.stack([y_channel, cb_channel, cr_channel])  # shape=[3,w,h]
    ycbcr_image_ = ycbcr_image_.transpose([1, 2, 0])  # shape=[w,h,3]
    return ycbcr_image_


def rgb2ycbcr(rgb_image):
    """
    convert rgb into ycbcr, input = RGB, if try cv.imread, please cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    """
    # 输入的是tensor类型转为ndarray
    if type(rgb_image) == torch.Tensor:
        rgb_image = np.array(rgb_image).astype(np.float32)

    # 如果是(3,128,128)类型，转为(128,128,3)
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        rgb_image = rgb_image.transpose([1, 2, 0])

    rgb_image = rgb_image.astype(np.float32)
    # 1：创建变换矩阵，和偏移量
    transform_matrix = np.array([[0.257, 0.504, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    # 标准不同
    # transform_matrix = np.array([[0.299, 0.587, 0.114],
    #                              [-0.169, -0.331, 0.5],
    #                              [0.5, -0.419, -0.081]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    # 2：遍历每个像素点的三个通道进行变换
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image


def DCT_Rearrange(data):
    """
    DCT块重排  eg. 3*128*128 -> （64*3） * 128 *128
    """
    m, n = data.shape[1], data.shape[2]
    data = np.array(data).astype(np.float64)

    fin = np.zeros((3 * 64, m // 8, n // 8))

    for each in range(3):  # 3个通道依次进行
        tmp, test = [], []
        img_s = []
        data_channel = data[each, :, :]
        for i in range(8):
            for j in range(8):
                for row in range(0, n, 8):
                    for col in range(0, n, 8):
                        tmp.append(data_channel[row + i, col + j])  # 得到一个按DCT块一个一个取的列表
        tmp = np.array(tmp)

        # 重排成64个图像
        for i in range(data_channel.size):
            if i % (data.shape[1] // 8) ** 2 == 0 and i != 0:
                img_s = np.array(img_s)
                test.append(img_s.reshape(data.shape[1] // 8, data.shape[2] // 8))
                img_s = []
                img_s.append(tmp[i])
            else:
                img_s.append(tmp[i])
                if i == data_channel.size - 1:
                    test.append(np.array(img_s).reshape(data.shape[1] // 8, data.shape[2] // 8))

        test = np.array(test)
        fin[each * 64:(each + 1) * 64, :, :] = test

    result = torch.from_numpy(fin)
    return result


def arrange(data):
    # ---- 复原成原始的DCT分块阵 ---- #
    if isinstance(data, torch.Tensor):
        data = np.array(data)

    w, h = data.shape[1] * 8, data.shape[2] * 8  # 重排复原后的宽高
    result = np.zeros([3, w, h])

    for each in range(3):
        data_channel = data[each * 64:(each + 1) * 64, :, :]
        data_channel = np.reshape(data_channel, [64 * data.shape[1] * data.shape[2]])
        tmplist = iter(data_channel)
        tmp = np.zeros((w, h))

        for i in range(8):
            for j in range(8):
                for row in range(0, w, 8):
                    for col in range(0, h, 8):
                        tmp[row + i, col + j] = next(tmplist)

        result[each, :, :] = tmp

    result = torch.from_numpy(result)
    return result


def catimg(data):
    for i in range(data.shape[0]):
        tmp = data[i]
        # tmp2 = cv2.idct(test_folder)
        tmp2 = tmp.astype(np.uint8)
        path = './test_folder/' + str(i) + '.png'
        cv2.imwrite(path, tmp2)


def DCT_Block(img_tensor, idct=False):
    """
    分块求DCT/IDCT矩阵
    :param img_tensor:  np or tensor
    :param idct: if idct
    :return: 返回相应矩阵
    """
    if len(img_tensor.shape) != 3 or img_tensor.shape[0] != 3:
        img_tensor = img_tensor.transpose([2, 0, 1])
    img_np = np.array(img_tensor).astype(np.float64)

    img_dct = np.zeros_like(img_np)

    for k in range(img_np.shape[0]):
        img_channel = img_np[k, :, :]
        temp = np.zeros_like(img_channel)
        m, n = img_channel.shape

        hdata = np.vsplit(img_channel, n / 8)  # 垂直分成高度度为8 的块
        for i in range(0, n // 8):
            blockdata = np.hsplit(hdata[i], m / 8)
            # 垂直分成高度为8的块后,在水平切成长度是8的块, 也就是8x8 的块
            for j in range(0, m // 8):
                block = blockdata[j]
                if not idct:
                    # Yb = block
                    Yb = cv2.dct(block.astype(np.float32))
                else:
                    Yb = cv2.idct(block.astype(np.float32))
                # iblock = cv2.idct(Yb)
                temp[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = Yb
        img_dct[k, :, :] = temp  # img_dct 是分块得到的dct值

    return img_dct


def calc_channel_all_sum(path, size=192):  # 计算均值的辅助函数，统计单张图像颜色通道和，以及像素数量
    data = to_dct_block(path, size)
    _, m, n = data.shape
    pixel_num = m * n
    channel_sum = data.sum(axis=(1, 2))  # 192个通道
    return channel_sum, pixel_num


def calc_channel_all_var(path, mean, size=192):  # 计算标准差的辅助函数
    data = to_dct_block(path, size)
    data, mean = np.array(data), np.array(mean)
    data = np.transpose(data, [1, 2, 0])
    channel_var = np.sum((data - mean) ** 2, axis=(0, 1))
    return channel_var


def calc_channel_mean(data, size=192):  # 计算均值的辅助函数，统计单张图像颜色通道和，以及像素数量
    data = np.array(data)
    mean, var = [], []
    _, m, n = data.shape
    # pixel_num = m * n
    for i in range(192):
        channel = data[i]
        mean_channel = np.mean(channel)
        var_channel = np.var(channel)
        mean.append(mean_channel)
        var.append(var_channel)
    # channel_sum = data.sum(axis=(1, 2))  # 192个通道
    # for i in range(192):
    #     data[i] = (data[i]-mean[i]) / var[i]
    return mean, var


def calc_channel_var(path, mean, size=192):  # 计算标准差的辅助函数
    data = to_dct_block(path, size)
    data, mean = np.array(data), np.array(mean)
    data = np.transpose(data, [1, 2, 0])
    channel_var = np.sum((data - mean) ** 2, axis=(0, 1))
    return channel_var


def cal_channel_max_min(data, size=192):
    data = data.cpu()
    data = np.array(data)
    data_max, data_min = [], []
    _, m, n = data.shape

    for i in range(192):
        channel = data[i]
        max_channel = np.max(channel)
        min_channel = np.min(channel)
        data_max.append(max_channel)
        data_min.append(min_channel)

    return data_max, data_min


def to_dct_block(path, size=192):
    img_bgr = cv2.imread(str(path))
    img_bgr = cv2.resize(img_bgr, (size, size))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR TO RGB
    img_ycbcr = rgb2ycbcr(img_rgb)  # rgb to ycbcr
    img_ycbcr = img_ycbcr.transpose([2, 0, 1])  # (128,128,3) -> (3,128,128)
    img_dct = DCT_Block(img_ycbcr)  # 分块DCT
    img_rebuild = DCT_Rearrange(img_dct)  # 分块后各自取出排列 （192，16，16）
    return img_rebuild


def separate_low_high(tensor):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    DCcomponent_Y = tensor[:, 0, :, :]
    DCcomponent_Cb = tensor[:, 64, :, :]
    DCcomponent_Cr = tensor[:, 128, :, :]
    DCcomponent = torch.stack([DCcomponent_Y,DCcomponent_Cb,DCcomponent_Cr],dim=1)
    low_y = tensor[:, 1:32, :, :]
    low_cb = tensor[:, 65:80, :, :]
    low_cr = tensor[:, 129:144, :, :]
    low = torch.cat([low_y, low_cb, low_cr], dim=1)

    high_y = tensor[:, 32:64, :, :]
    high_cb = tensor[:, 80:128, :, :]
    high_cr = tensor[:, 144:, :, :]
    high = torch.cat([high_y, high_cb, high_cr], dim=1)

    return DCcomponent, low, high


def re_low_high(tensor1, tensor2):
    low_y = tensor1[:, 0:16, :, :]
    low_cb = tensor1[:, 16:24, :, :]
    low_cr = tensor1[:, 24:32, :, :]
    high_y = tensor2[:, 0:48, :, :]
    high_cb = tensor2[:, 48:104, :, :]
    high_cr = tensor2[:, 104:160, :, :]
    fin = torch.cat([low_y, high_y, low_cb, high_cb, low_cr, high_cr], dim=1)
    return fin


def cal_mean_all_time():
    from pathlib import Path
    from tqdm import tqdm
    start = time.time()
    # train_path = Path(r'/opt/data/xiaobin/ABDH_UDH/ABDH/testimg')
    train_path = Path(r'/opt/data/mingjin/pycharm/Data/ABDH/train/img')
    img_all = list(train_path.rglob('*.jpg'))
    img_f = img_all[:10]
    n = len(img_f)
    NUM_THREADS = os.cpu_count()
    result = ThreadPool(NUM_THREADS).imap(calc_channel_all_sum(), img_f)  # 多线程计算
    channel_sum = np.zeros(192)
    cnt = 0
    pbar = tqdm(enumerate(result), total=n)
    for i, x in pbar:
        # x = np.array(x)
        channel_sum += np.array(x[0])
        cnt += np.array(x[1])
    mean = channel_sum / cnt

    result = ThreadPool(NUM_THREADS).imap(lambda x: calc_channel_var(*x), zip(img_f, repeat(mean)))
    channel_sum = np.zeros(192)
    pbar = tqdm(enumerate(result), total=n)
    for i, x in pbar:
        channel_sum += x
    var = np.sqrt(channel_sum / cnt)

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    import pandas as pd

    #
    # df = pd.DataFrame(mean)
    # df.to_csv("mean256.txt")
    # df = pd.DataFrame(var)
    # df.to_csv("var256.txt")

    ################# 老师测试 #####################
    path = 'test_folder/000000000081.jpg'
    img = cv2.imread(path)
    img = cv2.resize(img,(128,128))
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ycbcr = rgb2ycbcr(img)
    img_block_dct = DCT_Block(img_ycbcr)  # 分块求DCT
    img_arrange = DCT_Rearrange(img_block_dct)
    # img_arrange = torch.Tensor(img_arrange)
    img_sq, a,b = separate_low_high(img_arrange)
    print("111")
    # img_RGB = ycbcr2rgb(img_ycbcr)

    # data = to_dct_block(path, size=192)
    # mean, var = calc_channel_mean(data, 192)
    # max_, min_ = cal_channel_max_min(data, 192)
    # # for i in range(192):
    # #     data[i] = (data[i]-mean[i]) / (var[i] + 1e-8)
    # for i in range(192):
    #     data[i] = (data[i] - min_[i]) / (max_[i] - min_[i] + 1e-8)
    # print(torch.max(data), torch.min(data))
    ################# 老师测试 #####################

