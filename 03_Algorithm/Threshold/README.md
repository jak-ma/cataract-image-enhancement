# 阈值分割
## 阈值分割的基本原理
- 阈值分割是图像处理中最基本和常用的技术之一，其主要的原理是将图像像素分为两类
- 大于等于既定的阈值设置为目标像素，反之则设置为背景像素
## 阈值分割方法
使用不同的阈值分割算法，可以得到不同的二值化图像
- 全局阈值法
- 自适应阈值法
- 基于形态学的阈值分割方法（先进行一些形态学操作，再应用前两种方法进行阈值分割）
### 单阈值分割
```python
def test_threshold():
    img = cv2.imread('00_templates/guazi.png', 0)
    cv2.imshow('guazi', img)
    _, mask = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    ### TODO 除了 cv2.THRESH_BINARY，还有其他的方法，方法不同，应用也不同
    # 二值化、反二值化、截断、去背景等等
    cv2.imshow('guazi_mask', mask)
    cv2.waitKey(0)
```
### 自适应阈值分割
```python
def test_adaptiveThreshold():
    img = cv2.imread('00_templates/huahen.png', 0)
    cv2.imshow('guazi', img)
    mask = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 5)
    cv2.imshow('guazi_mask', mask)
    cv2.waitKey(0)
```
#### 拖动条的使用
`createTrackbar() 函数 | getTrackbarPos() 函数`
```python
def onChangeFunc(value, img):
    threshold = cv2.getTrackbarPos('ThresHoldBar', 'WindowsBar')
    maxval = cv2.getTrackbarPos('MaxValBar', 'WindowsBar')
    _, mask = cv2.threshold(img, threshold, maxval, cv2.THRESH_BINARY_INV)
    cv2.imshow('WindowsBar', mask)

if __name__ == '__main__':
    img = cv2.imread('00_templates/huahen.png', 0)
    cv2.namedWindow('WindowsBar')
    cv2.createTrackbar('ThresHoldBar', 'WindowsBar', 0, 255, lambda value:onChangeFunc(value, img))
    cv2.createTrackbar('MaxValBar', 'WindowsBar', 0, 255, lambda value:onChangeFunc(value, img))
    cv2.waitKey(0)
```
- `lambda表达式`的使用绕过了opencv对于回调函数 `onChange()`限制只能传入一个参数`value`的限制
