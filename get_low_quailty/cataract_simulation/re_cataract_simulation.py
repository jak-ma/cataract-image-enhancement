# 导入依赖
import cv2
import numpy as np
import random
import os
from scipy import ndimage
from PIL import Image, ImageEnhance

# 设置随机数种子
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

# 验证目录是否创建
def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

# 高斯核卷积 TODO 一种图像平滑技术，去掉小的噪点，边缘平滑，避免尖锐边缘
def gaussian(img):
    kernel_5x5 = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ])
    kernel_5x5 = kernel_5x5 / kernel_5x5.sum()  # TODO 注意这里使用 .sum() 和下面使用 .max() 的区别
    k5 = ndimage.convolve(img, kernel_5x5)
    return k5

# 高斯模糊在通道层面应用
def gaussian_blur_channel(img, sigma=[1.5, 1.2, 1.0]):
    channels = list(cv2.split(img))     # 使用list()转化是因为cv2.split(img)返回tuple类型不可修改
    for i in range(len(channels)):
        channels[i] = ndimage.gaussian_filter(channels[i], sigma[i])
    return cv2.merge(channels)

# 白内障眼底图像模拟
def cataract_simulation(filepath, maskpath, image_size):
    # 读取 img | mask
    img = cv2.imread(filepath, 1)
    img = cv2.resize(img, (image_size, image_size))
    mask, _, _ = cv2.split(cv2.imread(maskpath, 1))
    mask = mask / mask.max()
    
    # 计算 alpha*s(i, j)
    s = img.astype(np.float32) # 原始清晰眼底图像的亮度
    # alpha = random.uniform(0.7, 0.9)
    # 在全局加上不同颜色的不同衰减系数
    alpha_c = np.array([0.4, 0.75, 0.85])
    s_attention = alpha_c * s     # 得到式子中的第一部分，视网膜反射部分光
    
    # 计算 t(i, j)
    h, w ,c = img.shape
    wp = random.randint(int(-0.3*w), int(w*0.3))
    hp = random.randint(int(-0.3*h), int(h*0.3))
    transmap = np.ones(shape=[h, w])
    transmap[w//2+wp, h//2+hp] = 0
    transmap = gaussian(ndimage.distance_transform_edt(transmap))*mask
    transmap = transmap / transmap.max()
    tau = transmap  # 这里的话感觉还是和论文有点出入，不知道为啥这样效果好一点

    # L = 255.0
    # L也加上颜色衰减
    L_c = np.array([150, 200, 220], dtype=np.float32)
    tau = tau[:, :, np.newaxis]

    # 计算 panel
    panel = tau*(L_c-s_attention)
    beta = 0.6
    panel = beta * panel    # 乘上一个衰减系数

    sigma = [2.0, 1.5, 1.0]
    panel = gaussian_blur_channel(panel, sigma=sigma)
    panel = cv2.GaussianBlur(panel, (7, 7), 1.5)

    # 添加 "黄色" 偏置，为了增加白内障引起的黄褐色
    yellow_bias = tau*np.array([0, 60, 35])
    panel += yellow_bias

    mask = mask[:, :, np.newaxis]
    panel *= mask
    
    # 调试代码时所用
    # cv2.imwrite('get_low_quailty/test_image/panel.png', np.concatenate([panel, img], 1))

    # 加权合成
    img_A = 0.85*s_attention + panel
    img_A[img_A>255] = 255
    
    # 颜色增强
    c = random.uniform(0.9, 1.3)
    b = random.uniform(0.9, 1.0)
    e = random.uniform(0.9, 1.3)
    img_enhanced_color = Image.fromarray((img_A).astype('uint8'))

    enh_con = ImageEnhance.Contrast(img_enhanced_color).enhance(c)
    enh_bri = ImageEnhance.Brightness(enh_con).enhance(b)
    enh_col = ImageEnhance.Color(enh_bri).enhance(e)
    img_A= enh_col*mask

    return img_A, img