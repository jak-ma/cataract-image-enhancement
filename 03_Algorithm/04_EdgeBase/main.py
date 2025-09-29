import cv2

def test_sobel():
    img = cv2.imread('00_templates/guazi.png', cv2.COLOR_BAYER_BG2GRAY)
    cv2.imshow('guazi', img)

    x_grad = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    y_grad = cv2.Sobel(img, cv2.CV_32F, 0, 1)

    x_grad = cv2.convertScaleAbs(x_grad)
    cv2.imshow('x_grad', x_grad)

    y_grad = cv2.convertScaleAbs(y_grad)
    cv2.imshow('y_grad', y_grad)

    x_y = cv2.add(x_grad, y_grad, dtype=cv2.CV_16S)
    x_y = cv2.convertScaleAbs(x_y)
    cv2.imshow('x_y', x_y)
    
    cv2.waitKey(0)


if __name__ == '__main__':
    test_sobel()
