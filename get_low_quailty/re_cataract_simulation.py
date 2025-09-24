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
    transmap = 1 - transmap
    # L = 255.0
    # L也加上颜色衰减
    L_c = np.array([150, 200, 220])
    transmap = transmap[:, :, np.newaxis]

    panel = transmap*(L_c-s_attention)
    cv2.imwrite('get_low_quailty/test_image/panel.png', np.concatenate([panel, img], 1))

    img_A = s_attention + panel

    img_A[img_A>255] = 255
    mask = mask[:, :, np.newaxis]

    return img_A*mask, img

if __name__ == '__main__':
    print(os.getcwd())
    img_A, img = cataract_simulation('get_low_quailty/test_image/NL_001.png', 'get_low_quailty/test_image/NL_001_mask.png', 512)
    cv2.imwrite('get_low_quailty/test_image/re_method.png', np.concatenate([img_A, img], 1))