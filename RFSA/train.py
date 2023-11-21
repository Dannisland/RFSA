# -*- coding:utf-8 -*-
# !/usr/bin/python3

import argparse
import itertools
import datetime
import os

import kornia
import lpips
import numpy as np
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import torch
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import model
from model.ABDHmodel import Discriminator, AttentionNet, DCT, Encoder, \
     Decoder_cbam, Encoder2, AttentionNet_jpeg
from model.MyUNet.unet_models import UNet
from model.RevealNet import RevealNet
from utils import ReplayBuffer, LambdaLR, weights_init_normal, same_seeds
from datasets import ImageDataset
from utils_folder.ssim_loss import SSIM
from vgg_loss import VGGLoss
# from img_utils import separate_low_high
from utils_folder import focal_frequency_loss as FFL

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=130,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batchSize', type=int, default=5, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/opt/data/mingjin/pycharm/Data/ABDH',
                    # '/opt/data/mingjin/pycharm/Data/ABDH','/workshop/Datasets/COCO'
                    help='root directory of the dataset')
parser.add_argument('--lr_G', type=float, default=0.0001, help='initial learning rate for G')
parser.add_argument('--lr_D', type=float, default=0.0004, help='initial learning rate for G')
parser.add_argument('--updataD', type=int, default=1,
                    help='Update the generator three times, and the discriminator is updated once')
parser.add_argument('--size', type=int, default=128, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
# parser.add_argument('--output_nc', type=int, default=48, help='number of channels of output data')

parser.add_argument('--weight', type=int, default=1, help='realImage weights')
parser.add_argument('--LHloss', type=int, default=1, help='The ratio of low frequency to high frequency')
parser.add_argument('--DCloss', type=int, default=1, help='The ratio of DC frequency')
parser.add_argument('--ratio', type=int, default=1, help='The ratio of dct to img')
parser.add_argument('--dct_kernel', type=int, default=4, help='The size of the dct blockg')

parser.add_argument('--device_ID', type=str, default='cuda:2', help='the used device ID')
parser.add_argument('--vis_iter', type=int, default=100, help='Iterate a certain number of times for visualization')
parser.add_argument('--n_cpu', type=int, default=5, help='number of cpu threads to use during batch generation')
parser.add_argument('--exp', type=str, default='2022_2_14,use new encoder(freeze encoder) and new decoder(UDH)',
                    help='name of experiment')
opt = parser.parse_args()
tb_writer = SummaryWriter(logdir='./log/%s' % opt.exp)
print(opt)
same_seeds(1024)  # 固定 random seed 的函數，以便 reproduce。
device = torch.device(opt.device_ID)

def freeze_model(model, to_freeze_dict, keep_step=None):

    for (name, param) in model.named_parameters():
        if name in to_freeze_dict:
            param.requires_grad = False
        else:
            pass

    # # 打印当前的固定情况（可忽略）：
    # freezed_num, pass_num = 0, 0
    # for (name, param) in model.named_parameters():
    #     if param.requires_grad == False:
    #         freezed_num += 1
    #     else:
    #         pass_num += 1
    # print('\n Total {} params, miss {} \n'.format(freezed_num + pass_num, pass_num))

    return model



# -----Definition of variables----- #
# Networks
dct = DCT(N=opt.dct_kernel).to(device)
attentionModel = AttentionNet_jpeg().to(device)
# attentionModel = eca_resnet50().to(device)
# generator = GeneratorImage(2 * opt.input_nc + 2, opt.output_nc).to(device)
generator = UNet(opt.input_nc, opt.input_nc * opt.dct_kernel * opt.dct_kernel, bilinear=False, norm_layer=nn.InstanceNorm2d).to(device)
# generator = Encoder2(opt.input_nc, opt.input_nc * opt.dct_kernel * opt.dct_kernel, opt.dct_kernel).to(device)
# extractor = Decoder_cbam(opt.input_nc * opt.dct_kernel * opt.dct_kernel, opt.input_nc, opt.dct_kernel).to(device)
extractor = RevealNet(input_nc=opt.input_nc, output_nc=opt.input_nc, nhf=64, norm_layer=nn.InstanceNorm2d, output_function=nn.Sigmoid).to(device)

netD_A = Discriminator(opt.input_nc).to(device)
netD_B = Discriminator(opt.input_nc).to(device)

generator.apply(weights_init_normal)
extractor.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
# criterion_GAN = torch.nn.MSELoss()
criterion_Consistent = torch.nn.L1Loss().to(device)
criterion_L2 = torch.nn.MSELoss().to(device)
vgg_loss = VGGLoss(3, 1, False).to(device)
ffl = FFL.FocalFrequencyLoss(loss_weight=1.0, alpha=1.0).to(device)

# computeSSIM = SSIM().to(device)
# computeMSSSIM = MS_SSIM(1.0).to(device)
criterion_LPIPS = lpips.LPIPS(net='alex', verbose=False).to(device)

PSNR_Enc_epoch, SSIM_Enc_epoch, PSNR_Dec_epoch, SSIM_Dec_epoch = [], [], [], []
loss_G_iter, loss_G_GAN_iter, loss_G_Inconsistent_iter, loss_D_iter, loss_L1_iter = [], [], [], [], []
loss_G_epoch, loss_G_GAN_epoch, loss_G_Inconsistent_epoch, loss_D_epoch, loss_L1_epoch = [], [], [], [], []

loss_L1_A_img_epoch, loss_L1_B_img_epoch, loss_L1_A_dct_epoch, \
loss_L1_B_dct_epoch, loss_G_VGGenc_epoch, loss_G_VGGdec_epoch = [], [], [], [], [], []

# 加载checkpoint预训练模型
cp_path =  './output/' + opt.exp + '/46.pth' # './output/2022_2_11,use new encoder(encoder with Fca) and new decoder(UDH)/130.pth'
# cp_path = './output/2022_11_5_new_dowmsample kernel=4_VGG_+ffl/30.pth'
checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
# attentionModel.load_state_dict(checkpoint['model_attention'])
generator.load_state_dict(checkpoint['model_encoder'])
extractor.load_state_dict(checkpoint['model_decoder'])
# Hnet.load_state_dict(checkpoint['model_Hnet'])
# netD_A.load_state_dict(checkpoint['model_discA'])
# netD_B.load_state_dict(checkpoint['model_discB'])

generator = freeze_model(model=generator, to_freeze_dict=checkpoint['model_encoder'])


# Optimizers & LR schedulers
# optimizer_G = torch.optim.Adam(
#     itertools.chain(generator.parameters(), extractor.parameters(), attentionModel.parameters(), Hnet.parameters()),
optimizer_G = torch.optim.Adam(  # chain()可以把一组迭代对象串联起来，形成一个更大的迭代器,
    itertools.chain(filter(lambda p: p.requires_grad, generator.parameters()), extractor.parameters(), attentionModel.parameters()), lr=opt.lr_G,
    betas=(0.5, 0.999))

optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
target_real = torch.full((opt.batchSize, 1), 1.0, device=device)
target_fake = torch.full((opt.batchSize, 1), 0.0, device=device)
encoded_buffer = ReplayBuffer()
extracted_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [transforms.Resize([opt.size, opt.size], Image.BICUBIC),
               # transforms.RandomCrop(opt.size),
               # transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               ]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), pin_memory=True,
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

# logger = Logger(opt.n_epochs, len(dataloader))
###################################

# # ----- Training-----#
glob_step = checkpoint['glob_step']
star_epoch = checkpoint['epoch'] + 1
# ---------------从头训练
# glob_step = 0
# star_epoch = opt.epoch
print(glob_step, star_epoch)

# Hnet.train()  # 添加
attentionModel.train()
generator.train()
extractor.train()
netD_A.train()
netD_B.train()
for epoch in range(star_epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        glob_step += 1
        realImage, realSecret = batch['img'], batch['sec']  # [B,3,W,H]
        realImage, realSecret = realImage.to(device), realSecret.to(device)

        # jpeg_residual = batch['jpeg_residual']
        # jpeg_residual = jpeg_residual.to(device)

        # jnd = batch['jnd']
        # jnd = jnd.to(device)

        # # ----- Network Pipeline -----#
        # hidenet = Hnet(realSecret)
        # channel_a, spatial_a = attentionModel(jpeg_residual)

        realImage_inputs = dct(realImage).to(device)  # [B,192,W/8,H/8]

        rimax, rimin, realImage_inputs = dct.norm(realImage_inputs)  # realImage最大最小归一化处理
        rimax, rimin, realImage_inputs = rimax.to(device), rimin.to(device), realImage_inputs.to(device)

        encoded = generator(realSecret, realImage)  # [B,192,W/8,H/8]
        encoded = encoded + realImage_inputs
        # extracted = extractor(encoded)  # [B,3,W,H]

        # 逆归一化
        encoded2 = dct.renorm(encoded, rimax, rimin).to(device)

        # 还原图像
        realImage_outputs = dct.reverse(encoded2).to(device)  # [B,3,W,H]

        extracted = extractor(realImage_outputs)  # [B,3,W,H]

        # if glob_step % opt.updataD == 0 or glob_step == 1:
        #     # ----- Discriminator A -----#
        #     # Real loss
        #     pred_real = netD_A(realImage.detach())
        #     loss_D_real = criterion_GAN(pred_real, target_real)
        #
        #     # Fake loss
        #     fake_A = encoded_buffer.push_and_pop(realImage_outputs)  # encoded/encoded_idct
        #     # fake_A = encoded
        #     pred_fake = netD_A(fake_A.detach())
        #     loss_D_fake = criterion_GAN(pred_fake, target_fake)
        #
        #     # Total loss
        #     loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        #
        #     optimizer_D_A.zero_grad()
        #     loss_D_A.backward()
        #     optimizer_D_A.step()
        #     ###################################
        #
        #     # ----- Discriminator B -----#
        #     # Real loss
        #     # pred_real = netD_B(realSecret.detach())
        #     pred_real = netD_B(realSecret.detach())
        #     loss_D_real = criterion_GAN(pred_real, target_real)
        #
        #     # Fake loss
        #     fake_B = extracted_buffer.push_and_pop(extracted)  # extracted/extracted_idct
        #     # fake_B = extracted
        #     pred_fake = netD_B(fake_B.detach())
        #     loss_D_fake = criterion_GAN(pred_fake, target_fake)
        #
        #     # Total loss
        #     loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        #
        #     optimizer_D_B.zero_grad()
        #     loss_D_B.backward()
        #     optimizer_D_B.step()
        ###################################

        # ----- Generators A2B and B2A -----#
        # # GAN loss
        # pred_fake = netD_A(realImage_outputs)
        # loss_GAN_A2B = criterion_GAN(pred_fake, target_real)
        #
        # pred_fake = netD_B(extracted)
        # loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
        #
        # # Inconsistent loss
        # extract_from_cover = extractor(realImage_inputs)
        # # extract_from_cover_idct = TO_IDCT(extract_from_cover).to(device)
        # loss_Inconsistent = (0.6 / criterion_L2(extract_from_cover, extracted))

        # 其他Loss
        # ssim_loss_A = 1 - computeSSIM(realImage, realImage_outputs)
        # ssim_loss_B = 1 - computeSSIM(extracted, realSecret)
        # loss_L1_A = (0.16 * L1_A + 0.84 * ssim_loss_A) * 10
        # loss_L1_B = (0.16 * L1_B + 0.84 * ssim_loss_B) * 10

        #  ---- Loss A  ---- #
        # fflossA = ffl(realImage_outputs, realImage) * 10

        # vgg16loss
        vgg_loss_cover = vgg_loss(realImage)
        vgg_loss_encoded = vgg_loss(realImage_outputs)
        g_vgg_loss_enc = criterion_L2(vgg_loss_cover, vgg_loss_encoded)

        # 高低频块求Loss
        # DC_realImage, L_realImage, H_realImage = separate_low_high(realImage_inputs)
        # DC_encoded, L_encoded, H_encoded = separate_low_high(encoded)
        # L1_A_dct_DC = criterion_Consistent(DC_encoded, DC_realImage) * 60
        # L1_A_dct_l = criterion_Consistent(L_encoded, L_realImage) * 60
        # L1_A_dct_h = criterion_Consistent(H_encoded, H_realImage) * 3

        # L1 Loss
        L1_A = criterion_Consistent(realImage_outputs, realImage) * 50
        # L1_A_dct = L1_A_dct_DC * opt.DCloss + L1_A_dct_l * opt.LHloss + L1_A_dct_h
        L1_A_dct = criterion_Consistent(realImage_inputs, encoded) * 60

        # L2_A = criterion_L2(realImage_outputs, realImage) * 60
        loss_L1_A = L1_A_dct + L1_A * opt.ratio

        #  ---- Loss B ----  #
        # fflossB = ffl(extracted, realSecret) * 10

        vgg_loss_msg = vgg_loss(realSecret)
        vgg_loss_extracted = vgg_loss(extracted)
        g_vgg_loss_dec = criterion_L2(vgg_loss_msg, vgg_loss_extracted)

        L1_B = criterion_Consistent(extracted, realSecret) * 50
        # L2_B = criterion_L2(extracted, realImage) * 60

        loss_L1_B = L1_B * opt.weight

        # Total loss 试试加其他loss
        # loss_G = 0.1 * (loss_GAN_A2B + loss_GAN_B2A) + loss_L1_A + loss_L1_B + g_vgg_loss_enc
        loss_G = loss_L1_B + g_vgg_loss_dec + loss_L1_A + g_vgg_loss_enc  # + fflossB + fflossA#+ 0.1 * (loss_GAN_A2B + loss_GAN_B2A) + loss_Inconsistent * 0.1#

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        ###################################

        # ----- Record of indicators -----#
        with torch.no_grad():
            batch_enc_psnr = abs(kornia.losses.psnr_loss(realImage_outputs, realImage, 1))
            batch_dec_psnr = abs(kornia.losses.psnr_loss(extracted, realSecret, 1))

            PSNR_Enc_epoch.append(batch_enc_psnr.item())
            PSNR_Dec_epoch.append(batch_dec_psnr.item())

            # total loss(item)
            loss_G_iter.append(loss_G.item())
            # loss_G_GAN_iter.append((loss_GAN_A2B + loss_GAN_B2A).item())
            # loss_G_Inconsistent_iter.append(loss_Inconsistent.item())
            # loss_D_iter.append((loss_D_A + loss_D_B).item())
            loss_L1_iter.append((loss_L1_A + loss_L1_B).item())

            # total loss(epoch)
            loss_G_epoch.append(loss_G.item())
            # loss_G_GAN_epoch.append((loss_GAN_A2B + loss_GAN_B2A).item())
            # loss_G_Inconsistent_epoch.append(loss_Inconsistent.item())
            # loss_D_epoch.append((loss_D_A + loss_D_B).item())
            loss_L1_epoch.append((loss_L1_A + loss_L1_B).item())

            # L1A + L1B + VGG (epoch)
            loss_L1_A_img_epoch.append(L1_A.item())
            loss_L1_B_img_epoch.append(L1_B.item())
            loss_L1_A_dct_epoch.append(L1_A_dct.item())
            loss_G_VGGenc_epoch.append(g_vgg_loss_enc.item())  # add
            loss_G_VGGdec_epoch.append(g_vgg_loss_dec.item())  # add

        if glob_step % opt.vis_iter == 0 or glob_step == 1:
            tb_writer.add_scalar('iter_metric/loss_G', np.mean(loss_G_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_G_GAN', np.mean(loss_G_GAN_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_G_Inconsistent', np.mean(loss_G_Inconsistent_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_D', np.mean(loss_D_iter), glob_step)
            tb_writer.add_scalar('iter_metric/loss_L1', np.mean(loss_L1_iter), glob_step)

            # tb_writer.add_scalar('iter_metric/loss_L1_A_img', np.mean(loss_L1_A_img_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_L1_B_img', np.mean(loss_L1_B_img_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_L1_A_dct', np.mean(loss_L1_A_dct_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_L1_B_dct', np.mean(loss_L1_B_dct_iter), glob_step)
            #
            # tb_writer.add_scalar('iter_metric/loss_VGGenc', np.mean(loss_G_VGGenc_iter), glob_step)
            # tb_writer.add_scalar('iter_metric/loss_VGGdec', np.mean(loss_G_VGGdec_iter), glob_step)

            time = datetime.datetime.now()
            time_str = str(time).split('.')[0]
            print('---------- 当前时间 : %s ----------' % time_str)
            print('step:%d|G_loss:%.6s|loss_L1:%.6s' % (
                glob_step, np.mean(loss_G_iter), np.mean(loss_L1_iter)))

            loss_G_iter, loss_G_GAN_iter, loss_G_Inconsistent_iter, loss_D_iter, loss_L1_iter = [], [], [], [], []

        if glob_step % (10 * opt.vis_iter) == 0 or glob_step == 1:
            with torch.no_grad():
                # displayMasks = masksDisplay(attenMask.clone(), realImage.clone())
                # displayMasks = torch.cat([attenMask, attenMask, attenMask], dim=1)
                origin_img_grid = vutils.make_grid(realImage[:5, ...], nrow=5, normalize=True, scale_each=True)
                # attenMask_grid = vutils.make_grid(attentionMask[:5, ...], nrow=5)
                origin_sec_grid = vutils.make_grid(realSecret[:5, ...], nrow=5, normalize=True, scale_each=True)
                encoded_grid = vutils.make_grid(realImage_outputs[:5, ...], nrow=5, normalize=True, scale_each=True)
                extract_grid = vutils.make_grid(extracted[:5, ...], nrow=5, normalize=True, scale_each=True)
                # extract_from_cover_grid = vutils.make_grid(extract_from_cover[:5, ...], nrow=5, normalize=True,
                #                                            scale_each=True)

                tb_writer.add_image("image", origin_img_grid, glob_step)
                # tb_writer.add_image("attentionMask", attenMask_grid, glob_step)
                tb_writer.add_image("secret", origin_sec_grid, glob_step)
                tb_writer.add_image("target", encoded_grid, glob_step)
                tb_writer.add_image("extract", extract_grid, glob_step)
                # tb_writer.add_image("extract_from_cover", extract_from_cover_grid, glob_step)

    with torch.no_grad():
        tb_writer.add_scalar('epoch_metric/encoded_PSNR', np.mean(PSNR_Enc_epoch), epoch)
        tb_writer.add_scalar('epoch_metric/decoded_PSNR', np.mean(PSNR_Dec_epoch), epoch)

        tb_writer.add_scalar('epoch_metric/loss_G', np.mean(loss_G_epoch), epoch)
        # tb_writer.add_scalar('epoch_metric/loss_G_GAN', np.mean(loss_G_GAN_epoch), epoch)
        # tb_writer.add_scalar('epoch_metric/loss_G_Inconsistent', np.mean(loss_G_Inconsistent_epoch), epoch)
        # tb_writer.add_scalar('epoch_metric/loss_D', np.mean(loss_D_epoch), epoch)
        tb_writer.add_scalar('epoch_metric/loss_L1', np.mean(loss_L1_epoch), epoch)

        tb_writer.add_scalar('epoch_metric/loss_L1_A_img', np.mean(loss_L1_A_img_epoch), epoch)
        tb_writer.add_scalar('epoch_metric/loss_L1_B_img', np.mean(loss_L1_B_img_epoch), epoch)
        tb_writer.add_scalar('epoch_metric/loss_L1_A_dct', np.mean(loss_L1_A_dct_epoch), epoch)

        tb_writer.add_scalar('epoch_metric/loss_VGGenc', np.mean(loss_G_VGGenc_epoch), epoch)
        tb_writer.add_scalar('epoch_metric/loss_VGGdec', np.mean(loss_G_VGGdec_epoch), epoch)

        PSNR_Enc_epoch, SSIM_Enc_epoch, PSNR_Dec_epoch, SSIM_Dec_epoch = [], [], [], []
        loss_G_epoch, loss_G_GAN_epoch, loss_G_Inconsistent_epoch, loss_D_epoch, loss_L1_epoch = [], [], [], [], []
        loss_L1_A_img_epoch, loss_L1_B_img_epoch, loss_L1_A_dct_epoch, \
        loss_L1_B_dct_epoch, loss_G_VGGenc_epoch, loss_G_VGGdec_epoch = [], [], [], [], [], []

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Save models checkpoints
    state = {
        'model_attention': attentionModel.state_dict(),
        'model_encoder': generator.state_dict(),
        'model_decoder': extractor.state_dict(),
        # 'model_Hnet': Hnet.state_dict(),
        # 'model_discA': netD_A.state_dict(),
        # 'model_discB': netD_B.state_dict(),
        'epoch': epoch,
        'glob_step': glob_step
    }
    save_path = './output/%s/' % opt.exp
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, save_path + '%s.pth' % epoch)
    print("epoch:", epoch)
tb_writer.close()
