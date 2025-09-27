import cv2


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

if __name__ == '__main__':
    # test_threshold()
    # test_adaptiveThreshold()
    img = cv2.imread('00_templates/huahen.png', 0)
    cv2.namedWindow('WindowsBar')
    cv2.createTrackbar('ThresHoldBar', 'WindowsBar', 0, 255, lambda value:onChangeFunc(value, img))
    cv2.createTrackbar('MaxValBar', 'WindowsBar', 0, 255, lambda value:onChangeFunc(value, img))
    cv2.waitKey(0)