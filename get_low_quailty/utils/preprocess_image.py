import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import glob
import cv2
from PIL import ImageFile
### TODO 这个设置可以使得程序尽可能地加载损坏或者截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
from get_mask import imread, imwrite, preprocess
import numpy as np
from cataract_simulation.cataract_simulation import cataract_simulation

# ================================ #
#              全局配置             #
# ================================ #
# src 图片路径
image_root = r'images/original'
normal_path = r'normal'
cataract_path = r'cataract'
# dst 图片路径
save_source_path = r'images/kaggle/source_AB'
save_source_mask_path = r'images/kaggle/source_mask'
save_target_path = r'images/kaggle/target'
save_target_mask_path = r'images/kaggle/target_mask'
# 强制统一size
dsize = (512, 512)
# 同一退化可选参数
type_map = ['0010', '0100', '1000', '0110', '1010', '1100', '1110', '0001', '0001']
num_type = 16

# 默认只使用 TODO 模拟白内障 的退化效果 
def generate_type_list(num_type):
    return ['0001'] * num_type

def process(normal_image_list, cataract_image_list, num_type, save_source_path, save_source_mask_path, save_target_path, save_target_mask_path):
    # 确保 dst 目录安全存在
    os.makedirs(save_source_path, exist_ok=True)
    os.makedirs(save_source_mask_path, exist_ok=True)
    os.makedirs(save_target_path, exist_ok=True)
    os.makedirs(save_target_mask_path, exist_ok=True)

    # 生成 源域 图像对 [模拟白内障图像, 正常眼底图像]
    for image_path in normal_image_list:
        dst_source_base_image = os.path.basename(image_path).replace('.jpeg', '.png').replace('.jpg', '.png').replace('.tif', '.png')
        
        if os.path.exists(os.path.join(save_source_path, dst_source_base_image)):
            print('Current location has existed a image, skip...')
            continue

        try:
            img = imread(image_path)
            ### 生成 mask
            img, mask = preprocess(img)
            ### 白内障退化模拟
            img = cv2.cvtColor(cv2.resize(img, dsize), cv2.COLOR_BGR2RGB)
            mask = cv2.resize(mask, dsize)      # mask [h, w]
            r_mask = mask.astype(np.float32) / 255.0     # [0, 255] -> [0, 1]
            type_list = generate_type_list(num_type=num_type)

            for i, t in enumerate(type_list):
                dst_source_base = dst_source_base_image.split('.')[0]
                dst_img = os.path.join(save_source_path, f'{dst_source_base}-{i}.png')
                dst_mask_img = os.path.join(save_source_mask_path, f'{dst_source_base}-{i}.png')

                if t == '0001':
                    cataract_img, clear_img = cataract_simulation(img=img, mask=r_mask)
                    img_AB = np.concatenate([cataract_img, clear_img], 1)
                    cv2.imwrite(dst_img, img_AB)
                    cv2.imwrite(dst_mask_img, mask)

                else:
                    # 其他退化效果
                    pass
        except:
            print(f'There is an issue when generating the mask for the image ({image_path})')
            continue

    # 生成目标域图像 mask
    for image_path in cataract_image_list:
        dst_target_base_image = os.path.basename(image_path).replace('.jpeg', '.png').replace('.jpg', '.png').replace('.tif', '.png')
        target_img_path = os.path.join(save_target_path, dst_target_base_image)
        target_mask_path = os.path.join(save_target_mask_path, dst_target_base_image)

        if os.path.exists(os.path.join(target_img_path)):
            print('continue...')
            continue

        try:
            img = imread(image_path)
                
            img, mask = preprocess(img)
            img = cv2.resize(img, dsize)
            mask = cv2.resize(mask, dsize)
            imwrite(target_img_path, img)
            imwrite(target_mask_path, mask)

        except Exception as e:
            print(f'There is an issue when generating the mask for the image ({image_path}), error :{e}')
            continue

if __name__ == '__main__':
    # 创建自然图像路径列表
    normal_image_list = glob.glob(os.path.join(image_root, normal_path, '*.png'))
    normal_image_list += glob.glob(os.path.join(image_root, normal_path, '*.jpeg'))
    normal_image_list += glob.glob(os.path.join(image_root, normal_path, '*.tif'))
    # 创建真实白内障图像路径列表
    cataract_image_list = glob.glob(os.path.join(image_root, cataract_path, '*.png'))
    cataract_image_list += glob.glob(os.path.join(image_root, cataract_path, '*.jpeg'))
    cataract_image_list += glob.glob(os.path.join(image_root, cataract_path, '*.tif'))

    print('start preprocess')
    process(normal_image_list, cataract_image_list, num_type, save_source_path, save_source_mask_path, save_target_path, save_target_mask_path)
    print('preprocess end')
