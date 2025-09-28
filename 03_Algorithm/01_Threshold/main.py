import cv2
import numpy as np


def test_threshold():
    img = cv2.imread('00_templates/huahen.png', 0)
    cv2.imshow('guazi', img)
    _, mask = cv2.threshold(img, 180, 255, cv2.THRESH_TOZERO_INV)
    cv2.imshow('guazi_mask', mask)
    cv2.waitKey(0)

def test_adaptiveThreshold():
    img = cv2.imread('00_templates/huahen.png', 0)
    cv2.imshow('guazi', img)
    mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    cv2.imshow('guazi_mask', mask)
    cv2.waitKey(0)

def onChangeFunc(value, img):
    threshold = cv2.getTrackbarPos('ThresHoldBar', 'WindowsBar')
    maxval = cv2.getTrackbarPos('MaxValBar', 'WindowsBar')
    _, mask = cv2.threshold(img, threshold, maxval, cv2.THRESH_BINARY_INV)
    cv2.imshow('WindowsBar', mask)

def test_OTSU():
    img = cv2.imread('00_templates/guazi.png', 0)
    cv2.imshow('guazi', img)
    thresh, mask = cv2.threshold(img, 255, 180, cv2.THRESH_OTSU)
    cv2.imshow('mask_guazi', mask)
    print('cv:', thresh)

def test_OTSU_myself():
    img = cv2.imread('00_templates/guazi.png', 0)
    # 先将二维图像展平成 1 维数组
    img_ravel = np.array(img).ravel().astype(np.uint8)
    print(img_ravel.shape)
    # 总的像素数目
    PixSum = img_ravel.size
    print(PixSum)
    # 每个灰度值的像素数目
    PixCount = np.zeros(256)
    # 每个灰度值的像素数目所占的比例
    PixRate = np.zeros(256)
    # (1) 统计每个灰度值的像素数目
    for i in range(PixSum):
        PixCount[img_ravel[i]] += 1
    # (2) 统计每个灰度值的像素数目所占的比例
    PixRate = PixCount*1.0/PixSum
    # (3) 确定使得类间方差最大对应的阈值
    thresh = OTUS_find_max_var(PixRate)

    _, mask = cv2.threshold(img, thresh, 180, cv2.THRESH_BINARY)
    cv2.imshow('my_OTSU', mask)
    print('my_OTSU:', thresh)

def OTUS_find_max_var(PixRate):
    max_var = 0
    thresh = 0
    # 确定使得类间方差最大对应的阈值 1-255?
    for i in range(1, 256):
        # 背景
        u1_item = 0.0
        # 前景
        u2_item = 0.0
        # (1) 背景像素的比例
        w1 = np.sum(PixRate[:i])
        # (2) 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:
            for m in range(i):
                u1_item = u1_item + PixRate[m]*m
            u1 = u1_item*1.0 / w1
            for n in range(i, 256):
                u2_item = u2_item + PixRate[n]*n
            u2 = u2_item*1.0 / w2
            # (3) 计算类间方差 忘了就查一下公式 下面的式子是经过推导以后的结果
            var = w1*w2*np.power((u1-u2), 2)
            if max_var < var:
                max_var = var
                thresh = i
    return thresh

if __name__ == '__main__':
    # test_threshold()
    # test_adaptiveThreshold()
    # img = cv2.imread('00_templates/huahen.png', 0)
    # cv2.namedWindow('WindowsBar')
    # cv2.createTrackbar('ThresHoldBar', 'WindowsBar', 0, 255, lambda value:onChangeFunc(value, img))
    # cv2.createTrackbar('MaxValBar', 'WindowsBar', 0, 255, lambda value:onChangeFunc(value, img))
    test_OTSU()
    test_OTSU_myself()
    cv2.waitKey(0)