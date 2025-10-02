from abc import ABC, abstractmethod
import torch.utils.data as data 
from random import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser
    
    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        if opt.source_size_count == 1:
            if opt.load_size < 512:
                new_h = new_w = 286
            elif opt.load_size >= 512:
                new_h = new_w = 606
        else:
            if not opt.isTrain:
                new_h = new_w = opt.load_size
            else:
                if opt.load_size < 512:
                    new_h = new_w = random.choice([286, 306, 326, 346])
                elif opt.load_szie >= 512:
                    new_h = new_w = random.choice([526, 566, 606, 646])
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size*h // w
    
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    flip_vertical = random.random() > 0.5

    return {'load_size':new_h, 'crop_pos':(x, y), 'flip':flip, 'flip_vertical':flip_vertical}

# Gan 网络的输入图像大小要求是4的倍数，这个函数是在对图像尺寸进行微调
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base)*base)
    w = int(round(ow / base)*base)

    if h == oh and w == ow:
        return img
    
    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

# 打印图像的尺寸被调整以后的警告信息
def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4.")
        print(f"The loaded image size was ({ow}, {oh}), so it was adjusted to ({w}, {h})")

        __print_size_warning.has_printed = True

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size*oh/ow, crop_size))

    return img.resize((w, h), method)

# 感觉写的挺奇怪的？错误处理机制呢？
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size

    if (ow > tw or oh > th):
        return img.crop(x1, y1, x1+tw, y1+th)
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __flip_vertical(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def get_transform_six_channel(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    mask_transform_list = []

    if 'resize' in opt.preprocess:
        if params is None:
            osize = [opt.load_size, opt.load_size]
        else:
            load_size = params['load_size']
            osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
        mask_transform_list.append(transforms.Resize(osize, Image.NEAREST))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))
        mask_transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, Image.NEAREST)))
    
    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
            mask_transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))
            mask_transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
        mask_transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=Image.NEAREST)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
            mask_transform_list.append(transforms.RandomHorizontalFlip())

        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
            mask_transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    # 加入上下翻转
    # 这个参数同时控制水平翻转和垂直/上下翻转
    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomVerticalFlip())
            mask_transform_list.append(transforms.RandomVerticalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
            mask_transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip_vertical'])))
    
    if convert:
        transform_list += [transforms.ToTensor()]
        mask_transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list), transforms.Compose(mask_transform_list)