import cv2
import numpy as np

def test_canny():
    img = cv2.imread('00_templates/good.png', 0)
    cv2.imshow('good', img)

    l, u = auto_get_threshold(2)
    canny_img1 = cv2.Canny(img, l, u, L2gradient=True)
    cv2.imshow('canny_img1', canny_img1)

    # canny_img2 = cv2.Canny(img, 200, 360, L2gradient=True)
    # cv2.imshow('canny_img2', canny_img2)

    cv2.waitKey(0)

def auto_get_threshold(method=1):
    img = cv2.imread('00_templates/good.png', 0)
    img_mean = np.mean(np.array(img))
    # TODO sigma 可根据实际需要略微改动
    sigma = 0.33
    upper = 0
    lower = 0
    if method == 1:
        lower = int(max(0, (1.0-sigma)*img_mean))
        upper = int(min(255, (1.0+sigma)*img_mean))
    else:
        upper = int(2*img_mean)
        lower = int(upper/3)    # TODO 可微调的

    return lower, upper



if __name__ == '__main__':
    test_canny()