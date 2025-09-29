# Canny 边缘检测算法

## 1. 最优边缘准则

即理论上如何判断什么样的边缘识别效果才算是好的，具体需要依照以下3个原则：

- 最优检测：算法能够尽可能多地标识出图像中的实际边缘
- 最优定位准则：检测到的边缘点的位置距离实际边缘点的位置最近
- 检测点与边缘点一一对应
- 经验上比较合理的边缘指的是尽可能不发生很多像素堆叠在一起的边缘

## 2. Canny 算法的流程

1. 应用**高斯滤波**平滑图像（为了去除噪声）
2. 计算图像的**强度梯度**（intensity gradients）
3. 应用**非最大值抑制**（non-maximum suppression）技术来消除边缘**误检**（即消除本来不是边缘的部分）`(tip: 把粗线变成细线，留下细细的一条边)`
4. 应用双阈值法来确定可能存在的边界（即消除漏检）
5. 利用滞后技术来跟踪边界

## 重点参数：对双阈值的理解很重要
- minral:maxVal 常用比例为 1:2/1:3
```python
def test_canny():
    img = cv2.imread('00_templates/good.png', 0)
    cv2.imshow('good', img)

    canny_img1 = cv2.Canny(img, 10, 200, L2gradient=True)
    cv2.imshow('canny_img1', canny_img1)

    canny_img2 = cv2.Canny(img, 200, 360, L2gradient=True)
    cv2.imshow('canny_img2', canny_img2)

    cv2.waitKey(0)
```
- 上述代码中的两个阈值是随便取得，导致效果较差
- 下面记录两个工程上快速取值并得到效果较好的办法
```python
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
```
  



