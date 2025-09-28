# ROI Region of Interest(感兴趣区域)
## 图像相加
- 标量相加 | h, w, c相同 | 或者是广播机制
- 两幅图像加，达到饱和值截止
```python
def test_add():
    img2 = 134
    add_result = cv2.add(img, img2)
    cv2.imshow('add_result', add_result)
    cv2.waitKey(0)
```
## mask 的使用
- mask 里面只有 0（背景） 和 非0（感兴趣的|需要处理的）
- 作用：减少干扰 | 加快处理速度
```python
def test_mask():
    mask = np.zeros((400, 600), np.uint8)
    x, y, wp, hp = 100, 100, 200, 200
    mask[x:x+wp, y:y+hp] = 255
    cv2.imshow('mask_test' ,mask)
    cv2.waitKey(0)
```
## 图像基于权重相加
- `addWeighted_img = cv2.addWeighted(img1, alpha, img2, beta, gamma) 函数 `
- ` dst = saturate(src1*alpha + src2*beta + gamma) ` 
- 常见应用如 语义分割的效果图，两张图按照权重叠加在一起
- 类似的还有直接相减 `subtract_img = cv2.subtract(img1, img2) 函数 `
## 图像的位操作
先将图像中的每一个像素（如uint8类型，先转成二进制，然后进行相关的二进制操作）
- `cv2.bitwise_and(img1, img2, mask)`
- `cv2.bitwise_not(img1, mask)`
- `cv2.bitwise_or(img1, img2, mask)`
- `cv2.bitwise_xor(img1, img2, mask)`
## 图像的部分叠加
## 非规则的 ROI
