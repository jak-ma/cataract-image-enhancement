# 基本的输入输出
## imread() 函数读取，flag 参数

- `1` COLOR 模式
- `0` GRAYSCALE 模式
- `-1` UNCHANGED 模式
## imshow() 函数直接显示图片时，需要调整显示窗口

通过重新设置并命名一个和所显示图片的name相同的窗口

`main.py: test_saveImage()`

```py
def test_saveImage():
    img = cv2.imread('golang.jpg', 1) 
    cv2.namedWindow('de', cv2.WINDOW_NORMAL)	# TODO 这里与下文中imshow()的 str 部分相同
    cv2.resizeWindow('de', 800, 600)
    cv2.imshow('de', img)
    cv2.waitKey(0)	# 显示的图像保持在界面，按任意键退出
    cv2.destroyAllWindows()	# 关闭所有窗口
```

## imread() 函数读取中文命名的文件时，出现报错

```py
path = '中文.jpg'
img = cv2.imread(path, 1) 
```

`[ WARN:0@0.026] global loadsave.cpp:268 cv::findDecoder imread_('中文.jpg'): can't open/read file: check file path/integrity`

**原因**

1. 我们先去编辑这个 `.py` 文件，一般在 VS Code 里面会默认的以 `utf-8` 存储。运行时，python3源码读取 `.py` 文件，以字节形式读取 `path`，（此时 `path` 在 windows 系统下保存时使用的是 `utf-8` 存储编码规则）然后**使用 `utf-8` 的编码规则去转换成 `unicode` 字符串**（本质是 unicode 字符序列）
2. 然后这个 `path`（ `unicode` 字符串）传入到 imread() 函数中，但是 opencv 在Windows 系统下**不是直接使用 unicode API，而是使用 `gbk` 编码格式将其转换成字节形式**，记作字节1。
3. Windows 下的 NTFS 文件系统在存储数据时最终是以 UTF-16 (大端或小端) 来存储在磁盘中 (字节形式)，记作字节2。
4. 现在用 "字节1" 在文件系统中找 "字节2"，匹配不成功的原因：
   - 不同编码方式的基本编码单位不同（包括其长度和规则）
   - 不同编码方式下，同一种Unicode字符被解释的形式不同
5. 因为匹配失败，导致语句 `img = cv2.imread('中文.jpg', 1) ` 执行以后造成 **img = None**，引发后面的一系列报错。

**解决方案**	

有很多种，这个是比较完备和通用的一种

```py
def test_Chinese_path():
    img = cv2.imdecode(np.fromfile('中文.jpg', dtype=np.uint8), -1)     
    # TODO cv2.imdecode() 参数2 flag 必须带上
    cv2.namedWindow('de', cv2.WINDOW_NORMAL)	
    cv2.resizeWindow('de', 800, 600)
    cv2.imshow('de', img)
    cv2.waitKey(0)	
    cv2.destroyAllWindows()
```

保存路径中含有中文时，方法也是类似的

```py
def test_save_Chinese_path():
    img = cv2.imdecode(np.fromfile('中文.jpg', dtype=np.uint8), -1)
    cv2.namedWindow('de', cv2.WINDOW_NORMAL)	
    cv2.resizeWindow('de', 800, 600)
    cv2.imshow('de', img)
    save_path = '中文保存.jpg'
    cv2.imencode('.jpg', img)[1].tofile(save_path)
    cv2.waitKey(0)	
    cv2.destroyAllWindows()
```

