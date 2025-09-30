import cv2
import numpy as np

def test_erode():
    img = cv2.imread('00_templates/good.png', 0)
    cv2.imshow('good_img', img)
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    erode_img1 = cv2.erode(img, kernel1, iterations=1)
    cv2.imshow('erode_img1', erode_img1)
    erode_img2 = cv2.erode(img, kernel2, iterations=1)
    cv2.imshow('erode_img2', erode_img2)
    threshhold, thresh_img = cv2.threshold(img, 0, 180, cv2.THRESH_OTSU)
    cv2.imshow('thresh_img', thresh_img)
    erode_threshhold = cv2.erode(thresh_img, kernel1, iterations=1)
    cv2.imshow('erode_threshhold', erode_threshhold)
    cv2.waitKey(0)


def test_dilate():
    img = cv2.imread('00_templates/good.png', 0)
    cv2.imshow('good_img', img)
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    erode_img1 = cv2.dilate(img, kernel1, iterations=1)
    cv2.imshow('erode_img1', erode_img1)
    erode_img2 = cv2.dilate(img, kernel2, iterations=1)
    cv2.imshow('erode_img2', erode_img2)
    threshhold, thresh_img = cv2.threshold(img, 0, 180, cv2.THRESH_OTSU)
    cv2.imshow('thresh_img', thresh_img)
    erode_threshhold = cv2.dilate(thresh_img, kernel1, iterations=1)
    cv2.imshow('erode_threshhold', erode_threshhold)
    cv2.waitKey(0)


if __name__ == '__main__':
    # test_erode()
    test_dilate()
    cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)