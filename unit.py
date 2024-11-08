import cv2
import numpy as np
import gzip
import os


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def sort_contours(cnts, method):
    reverse = False
    i = 0

    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True

    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1

    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 外接矩阵
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))  # 打包排序

    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)  # 每个点的坐标和 x + y
    rect[0] = pts[np.argmin(s)]  # 返回和最小的点，即左上角的点
    rect[2] = pts[np.argmax(s)]  # 返回和最大的点，即右下角的点

    diff = np.diff(pts, axis=1)  # 每个点的坐标差值 y - x
    rect[1] = pts[np.argmin(diff)]  # 差值最小的点，即右上角的点
    rect[3] = pts[np.argmax(diff)]  # 差值最大的点，即左下角的点

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (top_left, top_right, bottom_right, bottom_left) = rect  # 找到四个坐标点(左上, 右上, 右下, 左下)

    widthA = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))  # 下边长
    widthB = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))  # 上边长
    maxWidth = max(int(widthA), int(widthB))  # 取最大边长( 宽 )

    heightA = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))  # 右边长
    heightB = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))  # 左边长
    maxHeight = max(int(heightA), int(heightB))  # 取最大边长( 高 )

    dst = np.array([  # 变换后的新坐标
        [0, 0],  # 左上
        [maxWidth - 1, 0],  # 右上
        [maxWidth - 1, maxHeight - 1],  # 右下
        [0, maxHeight - 1]], dtype="float32")  # 左下

    M = cv2.getPerspectiveTransform(rect, dst)  # 透视变换矩阵(将不规则的四边形映射到3*3矩形中)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))  # 将四边形在矩阵中 "拉正" 为矩形
    # 可以理解成一个不规则的四边形放到一个三维空间里，然后在三维空间里给它拉正为规则的四边形(不严谨！！！只是帮助理解！！！)

    return warped


def parse_mnist(minst_file_addr, flatten=False, one_hot=False):
    """解析MNIST二进制文件, 并返回解析结果
    输入参数:
        minst_file: MNIST数据集的文件地址. 类型: 字符串.
        flatten: bool, 默认Fasle. 是否将图片展开, 即(n张, 28, 28)变成(n张, 784)
        one_hot: bool, 默认Fasle. 标签是否采用one hot形式.

    返回值:
        解析后的numpy数组
    """
    minst_file_name = os.path.basename(minst_file_addr)  # 根据地址获取MNIST文件名字
    with gzip.open(filename=minst_file_addr, mode="rb") as minst_file:
        mnist_file_content = minst_file.read()
    if "label" in minst_file_name:  # 传入的为标签二进制编码文件地址
        data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8,
                             offset=8)  # MNIST标签文件的前8个字节为描述性内容，直接从第九个字节开始读取标签，并解析
        if one_hot:
            data_zeros = np.zeros(shape=(data.size, 10))
            for idx, label in enumerate(data):
                data_zeros[idx, label] = 1
            data = data_zeros
    else:  # 传入的为图片二进制编码文件地址
        data = np.frombuffer(buffer=mnist_file_content, dtype=np.uint8,
                             offset=16)  # MNIST图片文件的前16个字节为描述性内容，直接从第九个字节开始读取标签，并解析
        data = data.reshape(-1, 784) if flatten else data.reshape(-1, 28, 28)

    return data
