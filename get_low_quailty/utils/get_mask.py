import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

def get_transform(resize_or_crop, loadSizeX, loadSizeY, fineSize):
    transform_list = []
    if resize_or_crop == 'resize_and_crop':
        ### opencv|PIL 图像是(w, h) pytorch 是(h, w)
        osize = [loadSizeY, loadSizeX]
        transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif resize_or_crop == 'scale':
        osize = [loadSizeY, loadSizeX]
        transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
    transform_list += [transforms.ToTensor()]

    return transforms.Compose(transform_list)

### TODO 这个定义的好随便呀，感觉没有统一性
transform = get_transform('scale', loadSizeX=512, loadSizeY=512, fineSize=512)


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
        # bbox 边界框 一般格式 (x, y, w, h) 
        border = np.array((bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3], img.shape[0], img.shape[1]), dtype=np.int)
    
    image = image[border[0]:border[1], border[2]:border[3], ...]

    return image, border

# 获取 mask
def get_mask_BZ(img):
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = img
    
    # TODO 基于启发式函数 得到阈值
    threhold = np.mean(gray_img)/3-7
    # 二值化
    _, mask = cv2.threshold(gray_img, max(0, threhold), 1, cv2.THRESH_BINARY)
    # floodFill 操作要求传入的 mask 比 image 上下左右各多出一个像素，即大一圈
    add_mask = np.zeros((mask.shape[0]+2, mask.shape[1]+2), np.uint8)
    conv_mask = (1-mask).astype(np.uint8)   # mask 的二值进行反转

    ### TODO 这一步的目的是为了保留眼底图像中心区域的必要存在的小洞
    _, conv_mask, _, _ = cv2.floodFill(conv_mask, add_mask, (0, 0), (0), cv2.FLOODFILL_MASK_ONLY)
    _, conv_mask, _, _ = cv2.floodFill(conv_mask, add_mask, (mask.shape[1]-1, mask.shape[0]-1), (0), cv2.FLOODFILL_MASK_ONLY)

    mask = mask + conv_mask
    kernel = np.ones((20, 20))
    # 进行开运算
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    ### TODO 这种方式值得学习

    return mask

# TODO '_' 开始表示这是一个私有函数，内部使用
# 霍夫圆检测-从二值化的 mask 中找到一个圆的圆心和半径
def _get_center_radius_by_hough(mask):
    circles = cv2.HoughCircles((mask*255).astype(np.uint8), cv2.HOUGH_GRADIENT, 1, 1000, \
                               param1=5, param2=5, minRadius=min(mask.shape)//4, maxRadius=max(mask.shape)//2)
    center = circles[0, 0, :2]
    radius = circles[0, 0, 2]
    
    return center, radius

# 生成一个包含白色圆形的 mask，圆内为1，圆外为0
def _get_circle_by_center_bbox(shape, center, radius):
    center_mask = np.zeros(shape=shape).astype(np.uint8)
    center_tmp = (int(center[0]), int(center[1]))
    center_mask = cv2.circle(center_mask, center_tmp[::-1], int(radius), (1), -1)

    return center_mask

def get_mask(img):
    if img.ndim == 3:
        g_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:     # gary-> (H, W)
        g_img = img.copy()
    else:
        raise 'image dim is not 1 or 3'
    h, w = g_img.shape
    shape = g_img.shape[0:2]
    
    g_img = cv2.resize(g_img, (0, 0), fx=0.5, fy=0.5)
    tg_img = cv2.normalize(g_img, None, 0, 255, cv2.NORM_MINMAX)
    tmp_mask = get_mask_BZ(tg_img)      # 得到初步的mask
    
    center, radius = _get_center_radius_by_hough(tmp_mask)
    center = [center[1] * 2, center[0] * 2]     ### TODO 这里是需要进行审查一下的看看是否是符合 opencv 规则的传参顺序
    radius = int(radius * 2) - 3
    
    s_h = max(0, int(center[0] - radius))
    s_w = max(0, int(center[1] - radius))
    bbox = (s_h, s_w, min(h - s_h, 2 * radius), min(w - s_w, 2 * radius))
    
    tmp_mask = _get_circle_by_center_bbox(shape, center, radius)
    return tmp_mask, bbox, center, radius

def mask_image(img,mask):
    img[mask<=0,...]=0
    return img

# 将裁减以后得到的图像放到正方形 "黑布" 上
def supplemental_black_area(img,border=None):
    image=img
    if border is None:
        h,v=img.shape[0:2]
        max_l=max(h,v)
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)
        ### TODO 这个border的选择是基于一种启发式的方法来完成的
        border=(int(max_l/2-h/2),int(max_l/2-h/2)+h,int(max_l/2-v/2),int(max_l/2-v/2)+v,max_l)
    else:
        max_l=border[4]
        if image.ndim>2:
            image=np.zeros(shape=[max_l,max_l,img.shape[2]],dtype=img.dtype)
        else:
            image=np.zeros(shape=[max_l,max_l],dtype=img.dtype)    
    image[border[0]:border[1],border[2]:border[3],...]=img
    return image,border    


def preprocess(img):
    mask, bbox, _, _ =get_mask(img)
    r_img=mask_image(img,mask)

    r_img,r_border=remove_back_area(r_img,bbox=bbox)
    mask,_=remove_back_area(mask,border=r_border)
    r_img,sup_border=supplemental_black_area(r_img)
    mask,_=supplemental_black_area(mask,border=sup_border)
    return r_img,(mask*255).astype(np.uint8)
