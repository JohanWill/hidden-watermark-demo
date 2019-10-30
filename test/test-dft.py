import imutils
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ALPHA = 5
img_path = '../xiaozhi.jpg'
wm = '../logo.png'

def load_img(path):
    image = cv.imread(path)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    return image


def optimizeImageDim(image):
    """
    加快傅里叶变换速度，需要对图片尺寸做优化
    :param image:
    :return: 用0填充边框的原图最佳DTF尺寸
    """
    src_rows, src_cols, src_channels = image.shape

    # 将高和宽用0填充到2、3、5的乘积数
    addPixelRows = cv.getOptimalDFTSize(src_rows) # 3024 -> 3072
    addPixelCols = cv.getOptimalDFTSize(src_cols) # 4032 -> 4050

    # 在图像周围形成边框
    padded = cv.copyMakeBorder(image,0,addPixelRows-src_rows,0,addPixelCols-src_cols,cv.BORDER_CONSTANT)
    # plt.imshow(imutils.opencv2matplotlib(image))
    # plt.imshow(image)
    # plt.show()
    return padded


def split_img(image):
    r, g, b = cv.split(image)
    # 121 表示1*2 中的第一个
    # plt.subplot(131),plt.imshow(r,cmap = 'gray')
    # plt.subplot(132),plt.imshow(g,cmap = 'gray')
    # plt.subplot(133),plt.imshow(b,cmap = 'gray')
    # plt.show()

    # 为什么得到的是三张不同d灰度图呢？不是已经分离出R，G，B通道了吗？应该是分别是红色图，绿色图，蓝色图才对阿。
    # 原因是：当调用 imshow（R）时，是把图像的R，G，B三个通道的值都变为R的值，所以图像的颜色三通道值为（R，R，R）
    # 同理 imshow（G）和imshow（B）所显示d图像的颜色通道也依次为（G，G，G）和（B，B，B）。
    # 而 当三个通道d值相同时，则为灰度图。

    # cv.imshow("RED", r);  # 显示三通道的值都为R值时d图片
    # cv.imshow("GREEN", g);  # 显示三通道的值都为G值时d图片
    # cv.imshow("BLUE", b);  # 显示三通道的值都为B值时d图片
    # cv.waitKey(0);
    return r, g, b


def main():
    image = load_img(img_path).astype(np.float32) # (3072, 4050, 3)
    image = optimizeImageDim(image)
    # image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    # R, G, B = cv.split(image)

    # 离散傅里叶变换 , 时域 -> 频域
    # image_f_r = cv.dft(np.float32(R), flags=cv.DFT_COMPLEX_OUTPUT)
    # image_f_g = cv.dft(np.float32(G), flags=cv.DFT_COMPLEX_OUTPUT)
    # image_f_b = cv.dft(np.float32(B), flags=cv.DFT_COMPLEX_OUTPUT)
    image_f =cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT) # (3072, 4050, 3)

    h, w, c = image_f.shape
    # 图像中文本字符串的左下角坐标
    point = (int(h/4),int(w/4))
    # 画文字
    # color : 文字颜色
    # fontFace : 表示字体样式
    # fontScale : 文字缩放大小
    # thickness : 文本线条粗细
    # image_f = cv.putText(image_f,"FADADA",point,fontFace=cv.FONT_HERSHEY_DUPLEX,fontScale=6,color=(0,0,0),thickness=15)
    # image_f = cv.flip(image_f,-1)
    # image_f = cv.putText(image_f,"FADADA", point, fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=6,color=(0, 0, 0), thickness=15)
    # image_f = cv.flip(image_f, -1)

    # 反离散傅里叶变换
    # image_if_r = cv.idft(image_f_r, cv.DFT_SCALE | cv.DFT_REAL_OUTPUT,0)
    # image_if_g = cv.idft(image_f_g, cv.DFT_SCALE | cv.DFT_REAL_OUTPUT, 0)
    # image_if_b = cv.idft(image_f_b, cv.DFT_SCALE | cv.DFT_REAL_OUTPUT, 0)
    image_if = cv.idft(image_f, cv.DFT_REAL_OUTPUT,0)

    # magnitude_r = cv.magnitude(image_if_r[:, :, 0], image_if_r[:, :, 1])
    # magnitude_g = cv.magnitude(image_if_g[:, :, 0], image_if_g[:, :, 1])
    # magnitude_b = cv.magnitude(image_if_b[:, :, 0], image_if_b[:, :, 1])
    magnitude = cv.magnitude(image_if[:,:,0],image_if[:,:,1])
    # magnitude = cv.normalize(magnitude,0,255,cv.NORM_MINMAX)
    # magnitude = cv.cvtColor(magnitude_r,cv.COLOR_GRAY2RGB)

    plt.imshow(magnitude)
    plt.show()
if __name__ == '__main__':
    main()