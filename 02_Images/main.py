import cv2
import numpy as np

import matplotlib.pyplot as plt
# img = cv2.imread('path', cv2.IMREAD_GRAYSCALE) 
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def hist():
    img = cv2.imread('00_templates/golang.jpg', 1)
    # 使用 calHist() 函数计算直方图
    hist1  = cv2.calcHist(img, [0], None, [256], [0, 256])   
    # 参数 histSize 表示要分成几个桶来统计 参数 ranges 表示要统计的像素值范围
    hist2  = cv2.calcHist(img, [1], None, [256], [0, 256])   
    hist3  = cv2.calcHist(img, [2], None, [256], [0, 256]) 

    plt.subplot(221)
    plt.title('r_hist')
    plt.plot(hist1, color='r')

    plt.subplot(222)
    plt.title('g_hist')
    plt.plot(hist2, color='g')

    plt.subplot(223)
    plt.title('b_hist')
    plt.plot(hist3, color='b')
    plt.show()

if __name__ == '__main__':
    hist()