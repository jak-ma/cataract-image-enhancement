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

# 白内障眼底图像模拟
def cataract_simulation(filepath, maskpath, image_size):
    # 读取 img | mask
    img = cv2.imread(filepath, 1)
    img = cv2.resize(img, (image_size, image_size))
    mask, _, _ = cv2.split(cv2.imread(maskpath, 1))
    mask = mask / mask.max()
    
    # 模拟白内障相关的空间变化透射函数 t(i, j) | TODO 这里和论文有点出入
    h, w, c = img.shape
    wp = random.randint(int(-w*0.3), int(w*0.3))
    hp = random.randint(int(-h*0.3), int(h*0.3))
    transmap = np.ones(shape=[h, w])
    transmap[w//2+wp, h//2+hp] = 0
    transmap = gaussian(ndimage.distance_transform_edt(transmap))*mask
    transmap = transmap / transmap.max()

    # 模拟乘上衰减系数的眼底图像 a*s(i, j)
    randomR = random.choice([1, 3, 5, 7])   ### TODO 这里的高斯核参数部分是可以优化的
    randomS = random.randint(10, 30)
    funds_blur = cv2.GaussianBlur(img, (randomR, randomR), randomS)

    # 这里的panel的模拟 和 原论文不符合
    B, G, R = cv2.split(funds_blur)     # [h, w, 3] -> [h, w] * 3

    panel = cv2.merge([transmap*(B.max()-B), transmap*(G.max()-G), transmap*(R.max()-R)])
    cv2.imwrite('get_low_quailty/test_image/panel00.png', panel)
    panel_ratio = random.uniform(0.6, 0.8)

    # 加权合成
    sum_degrad = 0.8*funds_blur + panel*panel_ratio
    sum_degrad[sum_degrad>255] = 255

    # 颜色增强
    c = random.uniform(0.9, 1.3)
    b = random.uniform(0.9, 1.0)
    e = random.uniform(0.9, 1.3)
    img_enhanced_color = Image.fromarray((sum_degrad).astype('uint8'))

    enh_con = ImageEnhance.Contrast(img_enhanced_color).enhance(c)
    enh_bri = ImageEnhance.Brightness(enh_con).enhance(b)
    enh_col = ImageEnhance.Color(enh_bri).enhance(e)

    mask = mask[:, :, np.newaxis]   # TODO [h, w] -> [h, w ,1]
    img_A = enh_col*mask

    return img_A, img