# 形态学操作
## 腐蚀
- 用于缩小图像中物体的大小或者去除物体边界附近的像素点
- 通过在输入图像上滑动一个卷积核，如果卷积核完全包含在图像前景区域中，则将卷积核中心对应的前景区域中心像素设置为1 `(若是灰度图，这里则会去卷积核对应的前景区域里面的最小像素值)`，否则设置为0 
- kernel 的大小`(3, 3) (5, 5) (7, 7) (1, 9)...`和形状 `(矩形) (十字) (椭圆)...`对腐蚀的效果都有影响
- 不同的 kernel 有不同的功能 (可查)
```python
def test_erode():
    img = cv2.imread('00_templates/good.png')
    cv2.imshow('good_img', img)
    kernel1 = np.ones((3, 3), np.uint8)
    # TODO 可以设置 kernel 的形状和参数
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    erode_img1 = cv2.erode(img, kernel1, iterations=1)
    cv2.imshow('erode_img1', erode_img1)
    erode_img2 = cv2.erode(img, kernel2, iterations=1)
    cv2.imshow('erode_img2', erode_img2)
    cv2.waitKey(0)
```
## 膨胀
- 与腐蚀不同的是，卷积核覆盖的区域内有任何一个像素为1，则将中心像素设置为1
- 能够增加物体的大小，填充物体之间的空隙，或者连接分离的物体
- 膨胀操作可以使得物体更加的突出和粗化
```python
def test_dilate():
    img = cv2.imread('00_templates/good.png', 0)
    cv2.imshow('good_img', img)
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    dilate_img1 = cv2.dilate(img, kernel1, iterations=1)
    cv2.imshow('dilate_img1', dilate_img1)
    dilate_img2 = cv2.dilate(img, kernel2, iterations=1)
    cv2.imshow('dilate_img2', dilate_img2)
    threshhold, thresh_img = cv2.threshold(img, 0, 180, cv2.THRESH_OTSU)
    cv2.imshow('thresh_img', thresh_img)
    dilate_threshhold = cv2.dilate(thresh_img, kernel1, iterations=1)
    cv2.imshow('dilate_threshhold', dilate_threshhold)
    cv2.waitKey(0)
```
## 开运算 (Opening)
- 先腐蚀再膨胀
- 可以去除图像中的小型噪声点，平滑物体的边界， 分离相连的物体
## 闭运算 (Closing)
- 先膨胀再腐蚀
- 填充图像中的小洞，连接物体之间的空隙，平滑物体的边缘
## 顶帽算法
- 顶帽 = 原图 - 开运算后的图
- 主要用于增强图像中比周围区域亮的细节，特别是提取比结构元素小的明亮细节。
## 黑帽算法
- 黑帽 = 闭运算后的图 - 原图
- 主要用于增强图像中比周围区域暗的细节，特别适合于提取比结构元素小的暗色细节。
```python
    cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```