import glob
import os
import cv2
from PIL import ImageFile
### TODO 这个设置可以使得程序尽可能地加载损坏或者截断的图像
ImageFile.LOAD_TRUNCATED_IMAGES = True
from get_mask import imread, imwrite, preprocess


image_root = ''
save_root = ''


dsize = (512, 512)
def process(save_path, image_list):
    if not os.path.isdir(os.path.join(save_path, 'image')):
        os.mkdir(os.path.join(save_path, 'image'))
    
    if not os.path.isdir(os.path.join(save_path, 'mask')):
        os.mkdir(os.path.join(save_path, 'mask'))
    
    for image_path in image_list:
        ### 改用这个更具有跨平台适用性
        dst_image = os.path.basename(image_path).replace('.jpeg', '.png').replace('.jpg', '.png').replace('.tif', '.png')
        dst_image_path = os.path.join(save_path, 'image', dst_image)
        dst_mask_path = os.path.join(save_path, 'mask', dst_image)
        # 路径存在说明此处已经有文件了，跳过当前循环
        if os.path.exists(dst_image_path):
            print('continue...')
            continue

        try:
            img = imread(image_path)
            img, mask = preprocess(img)
            img = cv2.resize(img, dsize)
            mask = cv2.resize(mask, dsize)
            imwrite(img, dst_image_path)
            imwrite(mask, dst_mask_path)

        except:
            print(f'生成图片({image_path})的mask时出现问题')
            continue

### TODO 后面这个需要解决一下
def mkdir(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    mkdir(save_root)
    image_list = glob.glob(os.path.join(image_root, '*.png'))
    image_list += glob.glob(os.path.join(image_root, '*.jpeg'))
    image_list += glob.glob(os.path.join(image_root, '*.tif'))
    process(save_root, image_list)





