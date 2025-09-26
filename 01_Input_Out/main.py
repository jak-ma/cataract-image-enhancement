import cv2
import numpy as np

def test_saveImage():
    img = cv2.imread('../templates/golang.jpg', 1) 
    cv2.namedWindow('de', cv2.WINDOW_NORMAL)	# 这里与下文中imshow() 的n
    cv2.resizeWindow('de', 800, 600)
    cv2.imshow('de', img)
    cv2.waitKey(0)	# 显示的图像保持在界面，按任意键退出
    cv2.destroyAllWindows()	# 关闭所有窗口
    
def test_read_Chinese_path():
    img = cv2.imdecode(np.fromfile('../templates/中文.jpg', dtype=np.uint8), -1)     # TODO cv2.imdecode() 参数2 flag 必须带上
    cv2.namedWindow('de', cv2.WINDOW_NORMAL)	
    cv2.resizeWindow('de', 800, 600)
    cv2.imshow('de', img)
    cv2.waitKey(0)	
    cv2.destroyAllWindows()

def test_save_Chinese_path():
    img = cv2.imdecode(np.fromfile('../templates/中文.jpg', dtype=np.uint8), -1)
    cv2.namedWindow('de', cv2.WINDOW_NORMAL)	
    cv2.resizeWindow('de', 800, 600)
    cv2.imshow('de', img)
    save_path = '../templates/中文保存.jpg'
    cv2.imencode('.jpg', img)[1].tofile(save_path)
    cv2.waitKey(0)	
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_save_Chinese_path()