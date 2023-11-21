import torchvision
from torchvision import transforms

from utils_noise import blur,crop,jpeg_compression,gaussian
import torch
import torchvision.utils as vutils

def tensor2PIL(tensor):  # 将tensor-> PIL
    print(tensor.shape)
    unloader = transforms.ToPILImage()
    image = tensor.cuda(3).clone()
    image = image.squeeze(0)
    print(image.shape)
    image = unloader(image)
    return image

def jpeg(tensor):
    # jpeg压缩  tensor.shape = [ , , , ]
    tensor = tensor.cuda(3)
    jpeg_layer = jpeg_compression.JpegCompression(device='cuda:3')
    jpeg_img = jpeg_layer(tensor)
    # print(jpeg_img.shape)
    return jpeg_img

def gaussian_noise(tensor):
    # 高斯噪声  tensor.shape = [ , , , ]
    tensor = tensor.cuda(3)
    gaus_layer = gaussian.gaussian_kernel()
    gaus_img = gaus_layer(tensor)
    # print(gaus_img)
    return gaus_img

def blur_10(tensor):
    tensor = tensor.cuda(3)
    blur_layer = blur.Blur(tensor)
    # print(blur_layer.shape)
    return blur_layer

def Ran_crop(tensor,list = [(0.9,0.9),(0.9,0.9)]):
    tensor = tensor.cuda(3)
    crop_layer = crop.Crop(list[0],list[1])
    crop_img = crop_layer(tensor)
    Resize = transforms.Resize(size=(tensor.shape[2], tensor.shape[3]))
    crop_img_resize = Resize(crop_img)
    # print(crop_img_resize.shape)
    return crop_img_resize

def rotate_degree(tensor, degree):
    img = tensor2PIL(tensor)
    transform_1 = transforms.Compose([
        transforms.RandomRotation(degrees=(degree, degree), expand=True),
        transforms.ToTensor(),
    ])
    tensor = transform_1(img).unsqueeze(0).float()
    return tensor

def flip(tensor, flag): #flag=0 水平翻转，flag=1 垂直翻转
    img = tensor2PIL(tensor)
    if flag == 0:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
        ])
    elif flag == 1:
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=1),
            transforms.ToTensor(),
        ])
    tensor = transform(img).unsqueeze(0).float()
    return tensor



def main():
    # PIL图像->jepg压缩
    from PIL import Image, ImageOps
    img = Image.open('../datasets_dct/train/img/COCO_train2014_000000000025.jpg')
    tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)
    print(tensor.shape)
    vutils.save_image(tensor, 'a.png')

    # jpeg -> cpu
    # img2 = torch.randn([1, 3, 64, 64])
    # blur_layer = jpeg_compression.JpegCompression(device='CPU')
    # blured_img = blur_layer(tensor)

    # cuda
    img2 = torch.randn([1, 3, 64, 64]).cuda(3)
    # tensor_jpeg = jpeg(tensor)
    # tensor_gaussian = gaussian_noise(tensor)
    # tensor_blur = blur_10(tensor)

    range_list = [(0.9,0.9),(0.9,0.9)]
    tensor_crop = Ran_crop(tensor,range_list)


    vutils.save_image(tensor_crop, 'b.png')



if __name__ == "__main__":
    main()