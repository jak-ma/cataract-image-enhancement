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

    plt.subplot(311)
    plt.title('r_hist')
    plt.plot(hist1, color='r')

    plt.subplot(312)
    plt.title('g_hist')
    plt.plot(hist2, color='g')

    plt.subplot(313)
    plt.title('b_hist')
    plt.plot(hist3, color='b')
    plt.show()

def equalizeHist():
    img = cv2.imread('00_templates/golang.jpg', 0)
    cv2.namedWindow('origin_img', cv2.WINDOW_NORMAL)
    cv2.imshow('origin_img', img)

    hist1 = cv2.calcHist(img, [2], None, [256], [0, 256])
    plt.subplot(311)
    plt.title('original_img_hist')
    plt.plot(hist1, color='r')


    opencv_hist_img = cv2.equalizeHist(img)
    cv2.namedWindow('opencv_hist_img', cv2.WINDOW_NORMAL)
    cv2.imshow('opencv_hist_img', opencv_hist_img)

    hist2 = cv2.calcHist(opencv_hist_img, [2], None, [256], [0, 256])
    plt.subplot(312)
    plt.title('opencv_hist_img')
    plt.plot(hist2, color='g')


    hist3, img3 = myEqualizeHist()
    cv2.namedWindow('my_hist_img', cv2.WINDOW_NORMAL)
    cv2.imshow('my_hist_img', img3)

    plt.subplot(313)
    plt.title('my_hist_img')
    plt.plot(hist3, color='b')
    plt.show()

    cv2.waitKey(0)

def myEqualizeHist():
    img = cv2.imread('00_templates/golang.jpg', 0)
    h ,w = img.shape
    Nk = np.zeros(256, np.uint32)
    Pk = np.zeros(256, np.float32)
    for i in range(h):
        for j in range(w):
            Nk[img[i, j]]+=1
    Nk = Nk.astype(np.float32)
    Pk = Nk / (h*w)
    Sk = np.zeros(256, np.float32)
    for i in range(256):
        Sk[i] = Sk[i-1] + Pk[i]
    histf = np.zeros(256, np.uint8)
    histf = np.round(Sk*255).astype(np.uint8)
    
    for i in range(h):
        for j in range(w):
            img[i, j] = histf[img[i, j]]
    
    return histf, img

if __name__ == '__main__':
    # hist()
    # equalizeHist()
    ### 图像归一化
    img = cv2.imread('00_templates/golang.jpg', 0)
    cv2.namedWindow('origin_img', cv2.WINDOW_NORMAL)
    cv2.imshow('origin_img', img)
    min_max_img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    cv2.namedWindow('normlize_img', cv2.WINDOW_NORMAL)
    cv2.imshow('normlize_img', (min_max_img*255).astype(np.uint8))
    cv2.waitKey(0) 
    ###
    