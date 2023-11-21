# encoding: utf-8

import functools
import os
import torch
import torch.nn as nn
# from torchsummary import summary


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=None, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, output_function=output_function)

        self.model = unet_block
        # print(self.model)

        self.tanh = output_function == nn.Tanh
        if self.tanh:
            self.factor = 10 / 255
        else:
            self.factor = 1.0

        self.conv2img = nn.Conv2d(6, 3, kernel_size=1,stride=1)

    # def forward(self, input):
    #     return self.factor * self.model(input)

    def forward(self, rimg, simg, c_fimg):
        ens = self.factor * self.model(simg)
        enc = ens + rimg
        enc = torch.cat([enc, c_fimg], dim=1)
        out = self.conv2img(enc)


        return out


    # Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False,
                 norm_layer=None, use_dropout=False, output_function=nn.Sigmoid):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == None:
            use_bias = True
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            if output_function == nn.Tanh:
                up = [uprelu, upconv, nn.Tanh()]
            else:
                up = [uprelu, upconv, nn.Sigmoid()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            if norm_layer == None:
                up = [uprelu, upconv]
            else:
                up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downrelu, downconv]
                up = [uprelu, upconv]
            else:
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
            return torch.cat([x, self.model(x)], 1)


if __name__ == "__main__":
    # from torchviz import make_dot
    # from torchsummary import summary
    # device = torch.device("cuda:3")

    Hnet = UnetGenerator(input_nc=3, output_nc=3, num_downs=5, norm_layer=nn.InstanceNorm2d
                         , output_function=nn.Sigmoid)
    print(Hnet)
    inputs = torch.randn(2, 3, 128, 128)
    outputs = Hnet(inputs, inputs, inputs)
    print(outputs.shape)

    # summary(Hnet, (3, 128, 128))
    # # backbone = models.resnet50(pretrained=True)
    #
    # # print(backbone)
    # g = make_dot(outputs)
    # g.render('udh_encoder', view=False)
