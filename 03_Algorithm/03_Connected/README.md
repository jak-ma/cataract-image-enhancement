## 深拷贝和浅拷贝
- 很多时候我们对一张图像做过很多种处理，这张图已经失去了原本的样子，但是可能后续还需要使用它，所以就涉及拷贝的操作
- Deep Copy and Shallow Copy
- 关键区别在于内存指向是否和原来独立
```python
def test_copy():
    a = np.array([[1], [2], [3]])
    # 浅拷贝
    copy_b = a.view()
    copy_b[0] = [4]
    print(copy_b, "copy_b:",  a)
    print('--------------------')
    # 深拷贝
    deep_b = copy.deepcopy(a)
    deep_b[0] = [5]
    print(deep_b, "deep_b:", a)
```
## 添加噪声和生成图片
- `guassian = np.random.normal(mean, sigma, (img.shape)).astype(np.uint8)`
- mean为均值，variance 为标准差
```python
def test_guassian():
    img = np.zeros((480, 600, 3), np.uint8)
    img[:] = (156, 155, 154)
    mean = 0
    variance = 50
    sigma = math.sqrt(variance)

    guassian = np.random.normal(mean, sigma, (img.shape)).astype(np.uint8)
    bg_img = cv2.add(img, guassian)
    cv2.imshow('bg_img', bg_img)
    cv2.waitKey(0)
```
**生成图片**
- 通过画圆的案例学习
```python
def get_circle():
    bg_img = test_guassian()
    radius_list = [50, 51, 52, 60]
    center = (240, 300)
    range_x = (240-50, 240+50)
    range_y = (300-50, 300+50)
    for i, radius in enumerate(radius_list):
        img = bg_img.copy()
        center_x = np.random.randint(range_x[0], range_x[1])
        center_y = np.random.randint(range_y[0], range_y[1])
        center = (center_x, center_y)
        img_with_circle = cv2.circle(img, center, radius, (120, 120, 120), 5)
        cv2.imshow(f'bg_img_with_circle_{i}', img_with_circle)
    cv2.waitKey(0)
```
## 连通体检测
- 连通体的原理
- 分为4连通和8连通（如锯齿状的边缘）
- 梯形物体的检测-floodFill?
- 边缘填充的原理和实现
 ```python
def test_makeBorder():
    img = test_guassian()
    img_with_border = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    cv2.imshow('border', img_with_border)
    cv2.waitKey(0)
```
- `cv2.BORDER_CONSTANT` 常数填充方法，还有其他很多种填充方法
- 连通体检测
```python
def test_connectedComponents():
    img = cv2.imread('00_templates/link_test.png', 0)
    cv2.imshow('img', img)
    # TODO 传入参数 img 要求二值化图像
    num_labels, labels = cv2.connectedComponents(img, connectivity=8, ltype=cv2.CV_32S)
    print(f'连通体的个数是: {num_labels-1}')
    # 创建输出图像
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # TODO 数组从1开始是因为0是背景
    for i in range(1, num_labels):
        mask = labels == i
        color = np.random.randint(0, 255, size=3).tolist()
        # mask 中 bool 值为 True 的地方被赋值 color
        output[mask] = color
    cv2.imshow('img_c', output)
    cv2.waitKey(0)
```
