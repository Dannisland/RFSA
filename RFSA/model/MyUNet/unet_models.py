""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False, norm_layer=nn.InstanceNorm2d):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = out_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.C_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.C_down1 = (Down(64, 128, norm_layer))
        self.C_down2 = (Down(128, 256, norm_layer))
        self.C_down3 = (Down(256, 512, norm_layer))
        self.C_down4 = (Down(512, 1024 // factor, norm_layer))
        self.C_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.C_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.C_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.C_up4 = (Up(128, 64, bilinear, norm_layer))
        self.C_add = (AddOutCv(64, out_channels))
        # self.C_outc = (OutConv(64, out_channels))

        self.S_inc = (DoubleCv(n_channels, 64, norm_layer=norm_layer))
        self.S_down1 = (Down(64, 128, norm_layer))
        self.S_down2 = (Down(128, 256, norm_layer))
        self.S_down3 = (Down(256, 512, norm_layer))
        self.S_down4 = (Down(512, 1024 // factor, norm_layer))
        self.S_up1 = (Up(1024, 512 // factor, bilinear, norm_layer))
        self.S_up2 = (Up(512, 256 // factor, bilinear, norm_layer))
        self.S_up3 = (Up(256, 128 // factor, bilinear, norm_layer))
        self.S_up4 = (Up(128, 64, bilinear, norm_layer))
        # self.S_outc = (OutConv(64, out_channels))

    def forward(self, x, y):
        y1 = self.S_inc(y)
        y2 = self.S_down1(y1)
        y3 = self.S_down2(y2)
        y4 = self.S_down3(y3)
        y5 = self.S_down4(y4)
        y_up1 = self.S_up1(y5, y4, None)
        y_up2= self.S_up2(y_up1, y3, None)
        y_up3 = self.S_up3(y_up2, y2, None)
        y_up4 = self.S_up4(y_up3, y1, None)
        # logits = self.S_outc(y_up4)

        x1 = self.C_inc(x)
        x2 = self.C_down1(x1)
        x3 = self.C_down2(x2)
        x4 = self.C_down3(x3)
        x5 = self.C_down4(x4)
        x = self.C_up1(x5, x4, None)
        x = self.C_up2(x, x3, y_up1)
        x = self.C_up3(x, x2, y_up2)
        x = self.C_up4(x, x1, y_up3)
        x_out = self.C_add(x, y_up4)
        # logits = self.C_outc(x)

        return x_out




if __name__ == "__main__":
    # from torchviz import make_dot
    # from torchsummary import summary
    # device = torch.device("cuda:3")

    Hnet = UNet(3, 48, bilinear=False, norm_layer=nn.InstanceNorm2d)
    print(Hnet)
    inputs = torch.randn(2, 3, 256, 256)
    outputs = Hnet(inputs, inputs)
    print(outputs.shape)
