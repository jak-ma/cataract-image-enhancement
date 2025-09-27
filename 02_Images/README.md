# 图像的基本理论

## bmp的存储格式

bmp在磁盘中的存储形式

<img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20250926202214173.png" alt="image-20250926202207003" style="zoom: 50%;" />

数据组成如上图，其中在文件描述中，说明了颜色映射表的大小和图像矩阵的高|宽|通道数；但采用RGB标准的无压缩图没有颜色映射表图像矩阵在数据中存储时是从左到右，从上到下，数据中 最后一个Byte 表示图像的右上角

## cv.resize() 方法的特殊参数

<img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20250926205852345.png" alt="image-20250926205852279" style="zoom:50%;" />

最后一个参数 **`interpolation`**  换掉 默认的线性插值法 可能会提升一点点精度

不同的插值方法的应用场景不同，如有适合做放大的 resize，适合做缩小的 resize，适合要求处理速度快的 resize等

## 鼠标操作

鼠标操作是有它对应的应用和对应的函数的，具体的需要用时查询，如 **YOLO鼠标点击打标|悬停查看信息**

## 图像的灰度化

1. 灰度化的方法

   - 最大值法

   - 平均值法 （手动实现的时候需要注意**溢出**的问题）

   - 加权平均值法（根据人眼对不同颜色的敏感度的高低，给予 RGB 分量不同的权重，具体比例：`0.299-R  0.587-G  0.114-B`）

   - Gamma校正，公式简单可查

2. opencv 自带的两种方法

   ```py
   # 1
   gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   # 2
   img = cv2.imread('path', cv2.IMREAD_GRAYSCALE)
   ```

​	这两个函数是使用的灰度化方法一样的，是加权平均值法，使用 `cv2.diff()`  函数可能得到最大差值为1，why：因为自己实现的 加权平均值法时，float -> int ，会自动丢弃小数位，比如 1.6 -> 1

## 直方图

统计每种像素值的数量，会自动丢失空间信息

```py
def hist():
    img = cv2.imread('00_templates/golang.jpg', 1)
    # 使用 calHist() 函数计算直方图
    hist1  = cv2.calcHist(img, [0], None, [256], [0, 256])  
    # 参数 histSize 表示要分成几个桶来统计 参数 ranges 表示要统计的像素值范围
    hist2  = cv2.calcHist(img, [1], None, [256], [0, 256])
    plt.subplot(211)
    plt.title('r_hist')
    plt.plot(hist1, color='r')
    # plt.show()
    plt.subplot(212)
    plt.title('g_hist')
    plt.plot(hist2, color='g')
    plt.show()
```

## 直方图的均衡化

直方图的均衡化又称作直方图修平，就是把一已知灰度概率分布的图像，经过一种变换，使之演变成一幅具有均匀灰度概率分布的新图像。是一种非线性点运算，能够**增强图像的局部对比度**。

**算法流程**

1. 统计原图像素值的个数 `Nk`（像素值为k的像素的个数）
2. 计算它们的直方图概率 `Pk = Nk / N`
3. 计算累加直方图 `Sk = 求和(Pk)` 
4. 数组映射后对象取整 `Sk = int{(L-1)*Sk+0.5} = int{255*Sk+0.5}`
5. 根据映射关系对原图像素进行替换 `Rk -> Sk`

`main.py: myEqualizeHist()`

```py
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
```

## 图像的归一化

`cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)`

## 计算程序运行的时间
