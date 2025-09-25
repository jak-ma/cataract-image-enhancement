import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

# 定义 transform 加载器
def get_transform(resize_or_crop, loadSizeX, loadSizeY, fineSize):
    transform_list = []
    if resize_or_crop == 'resize_and_crop':
        osize = [loadSizeX, loadSizeY]
        transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'scale':
        osize = [loadSizeX, loadSizeY]
        transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

# 重新定义 cv2.imread() 函数
def imread(file_path, c=None):
    if c is None:
        im = cv2.imread(file_path)
    else:
        im = cv2.imread(file_path, c)
    if im is None:
        raise ValueError('Can not read image')
    
    if im.ndim == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    return im

def imwrite(file_path, img):
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img)

# 去掉背景区域 
def remove_back_area(img, bbox=None, border=None):
    image = img
    if border is None:
        # border 边界框 一般格式 (x, y, w, h) 
        border = np.array((bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3], img.shape[0], img.shape[1]), dtype=np.int)
    
    image = image[border[0]:border[1], border[2]:border[3], ...]

    return image, border

# 获取 mask
def get_mask_BZ(img):
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # 基于启发式函数 得到阈值
    threhold = np.mean(gray_img)/3-7
    # 二值化
    _, mask = cv2.threshold(gray_img, max(0, threhold), 1, cv2.THRESH_BINARY)
    # floodFill 操作要求传入的 mask 比 image 上下左右各多出一个像素，即大一圈
    add_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8)
    conv_mask = (1-mask).astype(np.unit8)   # mask 的二值进行反转

    ### TODO 这样处理得到的mask比较好
    _, conv_mask, _, _ = cv2.floodFill(conv_mask, add_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY)
    _, conv_mask, _, _ = cv2.floodFill(conv_mask, add_mask, (mask.shape[1]-1, mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)

    mask = mask + conv_mask
    kernel = np.ones((20, 20))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    ### TODO 这种方式值得学习

    return mask

# TODO '_' 开始表示这是一个私有函数，内部使用
def _get_center_radius_by_hough(mask):
    circles = cv2.HoughCircles((mask*255).astype(np.uint8), cv2.HOUGH_GRADIENT, 1, 1000, param1=5, param2=5, minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2)
    center = circles[0, 0, :2]
    radius = circles[0, 0, 2]

    return center, radius

# def _get_circle_by_center_bbox(shape, center, bbox, radius):


if __name__ == '__main__':
    print(get_transform('scale', loadSizeX=512, loadSizeY=512, fineSize=512))
