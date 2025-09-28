import cv2
import numpy as np 

def test_add():
    img2 = 134
    add_result = cv2.add(img, img2)
    cv2.imshow('add_result', add_result)
    cv2.waitKey(0)

def test_mask():
    mask = np.zeros((400, 600), np.uint8)
    x, y, wp, hp = 100, 100, 200, 200
    mask[x:x+wp, y:y+hp] = 255
    cv2.imshow('mask_test' ,mask)
    cv2.waitKey(0)

if __name__ == '__main__':
    img = cv2.imread('00_templates/guazi.png')
    # test_add()
    # test_mask()