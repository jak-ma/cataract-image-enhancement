import os



# 定义图像文件扩展名集合
IMG_EXTENSION = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def make_dataset(dir, max_dataset_size=float("inf"), extra_dir=None):
    images = []
    images2 = []

    assert os.path.isdir(dir), f'{dir} is not a valid directory'

    # 使用 sorted() 可以让遍历顺序按字母排序，保证可重现性
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                if extra_dir is not None:
                    path2 = os.path.join(extra_dir, fname)
                    images2.append(path2)
    if extra_dir is not None:
        return images[:min(max_dataset_size, len(images))], images2[:min(max_dataset_size, len(images2))]
    return images[:min(max_dataset_size, len(images))]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSION)