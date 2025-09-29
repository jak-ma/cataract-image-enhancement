import numpy as np
import copy
import math
import cv2

def test_copy():
    a = np.array([[1], [2], [3]])
    # 浅拷贝
    copy_b = a.view()
    copy_b[0] = [4]
    print(copy_b, "copy_b:",  a)
    print('--------------------')
    # 深拷贝
    deep_b = copy.deepcopy(a)
    deep_b[0] = [5]
    print(deep_b, "deep_b:", a)

def test_guassian():
    img = np.zeros((480, 600, 3), np.uint8)
    img[:] = (156, 155, 154)
    mean = 0
    variance = 50
    sigma = math.sqrt(variance)

    guassian = np.random.normal(mean, sigma, (img.shape)).astype(np.uint8)
    bg_img = cv2.add(img, guassian)
    cv2.imshow('bg_img', bg_img)
    cv2.waitKey(0)

    return bg_img

def get_circle():
    bg_img = test_guassian()
    radius_list = [50, 51, 52, 60]
    center = (240, 300)
    range_x = (240-50, 240+50)
    range_y = (300-50, 300+50)
    for i, radius in enumerate(radius_list):
        img = bg_img.copy()
        center_x = np.random.randint(range_x[0], range_x[1])
        center_y = np.random.randint(range_y[0], range_y[1])
        center = (center_x, center_y)
        img_with_circle = cv2.circle(img, center, radius, (120, 120, 120), 5)
        cv2.imshow(f'bg_img_with_circle_{i}', img_with_circle)
    cv2.waitKey(0)

def test_makeBorder():
    img = test_guassian()
    img_with_border = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    cv2.imshow('border', img_with_border)
    cv2.waitKey(0)

def test_connectedComponents():
    img = cv2.imread('00_templates/link_test.png', 0)
    cv2.imshow('img', img)
    # TODO 传入参数 img 要求二值化图像
    num_labels, labels = cv2.connectedComponents(img, connectivity=8, ltype=cv2.CV_32S)
    print(f'连通体的个数是: {num_labels-1}')
    # 创建输出图像
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # TODO 数组从1开始是因为0是背景
    for i in range(1, num_labels):
        mask = labels == i
        color = np.random.randint(0, 255, size=3).tolist()
        # mask 中 bool 值为 True 的地方被赋值 color
        output[mask] = color
    cv2.imshow('img_c', output)
    cv2.waitKey(0)


if __name__ == '__main__' :
    # test_copy()
    # test_guassian()
    # get_circle()
    # test_makeBorder()
    test_connectedComponents()
