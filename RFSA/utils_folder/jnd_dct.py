import cv2
import numpy as np
import torch

# from utils_DCT import ycbcr2rgb

y_quantization_table = [16, 11, 10, 16, 24, 40, 51, 61,
                  12, 12, 14, 19, 26, 58, 60, 55,
                  14, 13, 16, 24, 40, 57, 69, 56,
                  14, 17, 22, 29, 51, 87, 80, 62,
                  18, 22, 37, 56, 68, 109, 103, 77,
                  24, 35, 55, 64, 81, 104, 113, 92,
                  49, 64, 78, 87, 103, 121, 120, 101,
                  72, 92, 95, 98, 112, 100, 103, 99]
y_quantization_table = np.array(y_quantization_table).reshape(8, 8)

cbcr_quantization = [17, 18, 24, 47, 99, 99, 99, 99,
                     18, 21, 26, 66, 99, 99, 99, 99,
                     24, 26, 56, 99, 99, 99, 99, 99,
                     47, 66, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99,
                     99, 99, 99, 99, 99, 99, 99, 99]
cbcr_quantization = np.array(cbcr_quantization).reshape(8, 8)


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
    dc_block = []
    partition = []

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
                    Fb = Yb[0][0]
                else:
                    Yb = cv2.idct(block.astype(np.float32))
                    Fb = Yb[0][0]
                # iblock = cv2.idct(Yb)
                temp[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = Yb
                partition.append(Yb)
                dc_block.append(Fb)

        img_dct[k, :, :] = temp  # img_dct 是分块得到的dct值

    return img_dct, dc_block, partition


def block2img(lists):
    size = int(np.sqrt(len(lists)) * 8)
    re_img = np.zeros([size, size])
    st_w = 0
    cnt = 0
    for i in range(int(size / 8)):
        st_h = 0
        for j in range(int(size / 8)):
            re_img[st_w: st_w + 8, st_h: st_h + 8] = lists[cnt]
            st_h = st_h + 8
            cnt = cnt + 1
        st_w = st_w + 8
    return re_img


def cal_global_dct(img):
    # 计算全图的dct值
    global_dct_y = cv2.dct(img[0])
    global_dct_cb = cv2.dct(img[1])
    global_dct_cr = cv2.dct(img[2])
    global_img_dct = np.stack([global_dct_y, global_dct_cb, global_dct_cr], axis=0)
    return global_img_dct


def calculate_luminance_quantization_table(img, dc_block):
    """
    计算亮度掩膜量化表
    img: ycbcr空间值
    dc_block： size/8*size/8个8*8块的dc系数
    """
    channel, size = img.shape[0], img.shape[1]
    block_num = int(size / 8 * size / 8)
    global_img_dct = cal_global_dct(img)

    # 计算各通道的全局dc值，global_y_dc = F(0,0)
    # global_y_dc = global_img_dct[0][0][0]
    # global_cb_dc = global_img_dct[1][0][0]
    # global_cr_dc = global_img_dct[2][0][0]
    #
    # dc_block_y = dc_block[0:block_num]
    # dc_block_cb = dc_block[block_num:block_num * 2]
    # dc_block_cr = dc_block[block_num * 2:block_num * 3]

    # 计算亮度掩膜的量化表
    aT = 0.649
    Lum_mask_quantization_list = []
    for i in range(channel):
        global_dc = global_img_dct[i][0][0]  # 计算各通道的全局dc值
        dc_block_ = dc_block[block_num * i:block_num * (i + 1)]

        for j in range(block_num):
            if i == 0:
                quantization = y_quantization_table  # 选择量化表
            else:
                quantization = cbcr_quantization
            t_k = quantization * np.power((dc_block_[j] / global_dc), aT)
            Lum_mask_quantization_list.append(t_k)

    return Lum_mask_quantization_list


def calculate_contrast_quantization_table(luminance_table, block_list):
    """
    计算对比度掩膜量化表
    luminance_table: 亮度掩膜系数表
    block_list： 切分的8*8块集合
    """
    block_num = len(luminance_table)
    m = np.zeros([block_num, 8, 8])
    for k in range(block_num):
        for i in range(8):
            for j in range(8):
                if i == 0 and j == 0:
                    w = 0
                else:
                    w = 0.7
                f_ = (abs(block_list[k][i][j]) ** w) * luminance_table[k][i][j] ** (1 - w)
                m[k][i][j] = max(f_, luminance_table[k][i][j])

    return m


def calculate_frequency_quantization_table(contrast_table):
    """
    计算频域掩膜量化表
    contrast_table: 对比度掩膜系数表
    quantization： 最初的量化表
    """
    beta = 4
    sum = 0
    p = np.zeros([len(contrast_table) // 256, 8, 8])
    d = np.zeros([len(contrast_table), 8, 8])
    for k in range(len(contrast_table)):
        if k <= 255:
            quantization = y_quantization_table
        else:
            quantization = cbcr_quantization
        for i in range(8):
            for j in range(8):
                d[k][i][j] = contrast_table[k][i][j] / quantization[i][j]

    for c in range(len(contrast_table) // 256):
        for i in range(8):
            for j in range(8):
                for k in range(len(contrast_table)):
                    d_ = abs(d[k][i][j]) ** beta
                    sum = d_ + sum
                p[c][i][j] = sum ** (1 / beta)
    return p


def cal_jnd_dct(img):
    # img输入需要转换为ycbcr
    img_ycbcr = rgb2ycbcr(img)
    img_ycbcr = img_ycbcr.transpose([2, 0, 1])

    img_block_dct, dc_block, qiefenlist = DCT_Block(img_ycbcr)

    # 计算亮度掩膜的量化表
    lum_mask_quantization_list = calculate_luminance_quantization_table(img_ycbcr, dc_block)
    # 计算对比度掩膜的量化表
    contrast_mask_quantization_list = calculate_contrast_quantization_table(lum_mask_quantization_list, qiefenlist)
    # 计算频域掩膜的量化表
    fre_mask_quantization_list = calculate_frequency_quantization_table(contrast_mask_quantization_list)

    return fre_mask_quantization_list


if __name__ == "__main__":

    size = 128
    block_num = int(size / 8 * size / 8)

    # 读入img 调整为128*128大小
    img = cv2.imread("../testimg/baboon.png", 0)
    img = cv2.resize(img, [size, size])

    # 转为ycbcr, 3*128*128
    img = np.stack([img, img, img], axis=0)
    img = img.astype(np.float)
    img_ycbcr = rgb2ycbcr(img)
    img_ycbcr = img_ycbcr.transpose([2, 0, 1])
    # img_ycbcr = img_ycbcr - 16

    # 计算整张图的dct值
    global_img_dct = cal_global_dct(img_ycbcr)

    # 计算分块dct（整张图）、各个分块的dc值、分块dct（8*8的list）
    img_block_dct, dc_block, qiefenlist = DCT_Block(img_ycbcr)
    qiefen_y = qiefenlist[0:block_num]
    fre = cal_jnd_dct(img)
    #
    # # 计算亮度掩膜的量化表
    # lum_mask_quantization_list = calculate_luminance_quantization_table(img_ycbcr, dc_block)
    # contrast_mask_quantization_list = calculate_contrast_quantization_table(lum_mask_quantization_list, qiefenlist)
    # fre_mask_quantization_list = calculate_frequency_quantization_table(contrast_mask_quantization_list)

    # 反量化
    quantification_result_y = []

    for i in range(block_num):
        q_img = np.round(qiefen_y[i] / lum_mask_quantization_list[i])
        iq_img = q_img * lum_mask_quantization_list[i]
        quantification_result_y.append(iq_img)

    # 还原至128*128
    timg = block2img(quantification_result_y)
    # ttt = cv2.normalize(timg, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite("img_re1.jpg", ttt)

    timg_3c = np.stack([timg, img_block_dct[1], img_block_dct[2]], axis=0)
    img_block_idct, _, _ = DCT_Block(timg_3c, idct=True)
    img_reycbcr = np.stack([img_block_idct[0], img_ycbcr[1], img_ycbcr[2]], axis=0)

    img_rergb = ycbcr2rgb(img_block_idct)
    cv2.imwrite("img_re.jpg", img_rergb)

    print("1")
