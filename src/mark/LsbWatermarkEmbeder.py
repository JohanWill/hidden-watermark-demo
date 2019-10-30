# 基于LSB最低有效位算法添加数字水印

"""
TODO: 1.编码文本的时候，应该将文本信息如：姓名+身份证号 转成固定长度的密码格式比如hash串
        然后用循环的方式写满整个图片，而不是像现在只写一次[文本长度+文本内容]在图片左上角
      2.嵌入图片
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class SteganographyException(Exception):
    pass


class LsbWatermarkEmbeder():
    def __init__(self, image):
        # 传入的图片
        self.image = image
        # 传入图片的高，宽，通道数
        self.height, self.width, self.channels = image.shape
        # 传入图片的尺寸
        self.size_ = self.width * self.height

        # 一个8位二进制数的列表
        # 1->00000001, 2->00000010, 4->000001000 ...
        # 将用来做位操作
        self.maskONEValues = [1, 2, 4, 8, 16, 32, 64, 128]
        self.maskONE = self.maskONEValues.pop(0)

        # 254 = 255 - 2^0 -> 11111110,
        # 253 = 255 - 2^1 -> 11111101,
        # 251 = 255 - 2^2 -> 11111011
        # ...
        self.maskZEROValues = [254, 253, 251, 247, 239, 223, 191, 127]
        self.maskZERO = self.maskZEROValues.pop(0)

        # 当前图片的高，宽，通道位置
        self.current_width = 0
        self.current_height = 0
        self.current_channel = 0

    @property
    def row(self):
        return self.height
    @property
    def col(self):
        return self.width
    @property
    def channel(self):
        return self.channels
    @property
    def size(self):
        return self.size_


    def put_binary_value(self,bits):
        """
        把二进制数放进图像中
        :param bits: 比如16位二进制数 0000000000000110
        :return:
        """
        for bit in bits:
            # print(self.current_height, self.current_width,self.current_channel)

            # 把当前点的像素值(B,G,R)变成list [B,G,R]
            val = list(self.image[self.current_height, self.current_width])
            # 进行位运算
            # and or 符号 [&] [|] 两侧若都是数值，那么进行位运算
            # & 按位与运算符：参与运算的两个值,如果两个相应位都为1,则该位的结果为1,否则为0
            # | 按位或运算符：只要对应的二个二进位有一个为1时，结果位就为1。
            if int(bit) == 1:
                # 如果当前位是1,把B通道的数值和2进行 | 位运算
                val[self.current_channel] = int(val[self.current_channel]) | self.maskONE
            else:
                # 如果当前位是0，把B通道的数值和254进行 & 位运算
                val[self.current_channel] = int(val[self.current_channel]) & self.maskZERO
            self.image[self.current_height, self.current_width] = tuple(val)
            # 将当前像素位置和通道下标分别加1
            # 因为我们移动了游标，所以其实只是传进来的二进制数的每一位的值
            # 分别跟每个像素的单通道数值的二进制最低位进行位运算
            # 一个像素点，有三个通道，所以可以放三个值
            self.next_slot()



    def next_slot(self):
        """
        移动到下一个像素点
        :return:
        """
        # 如果当前是最后一个通道，那么重新置为第一个通道,否则 +1
        if self.current_channel == self.channels - 1:
            self.current_channel = 0
            # 如果当前是行的最后一个像素，那么重新置为行的第一个像素,否则 +1
            if self.current_width == self.width - 1:
                self.current_width = 0
                # 如果当前是列的最后一个像素，那么重新置为列的第一个像素,否则 +1
                if self.current_height == self.height - 1:
                    self.current_height = 0
                    if self.maskONE == 128:
                        raise SteganographyException("No available slot remaining (image filled)")
                    else:
                        self.maskONE = self.maskONEValues.pop(0)
                        self.maskZERO = self.maskZEROValues.pop(0)
                else:
                    self.current_height += 1
            else:
                self.current_width += 1
        else:
            self.current_channel += 1


    def binary_value(self,val,bit_size):
        # 返回一个整数 int 或者长整数 long int 的二进制表示。
        # 从2开始截取是因为前面的0b只是前缀
        # 0b110 -> 110
        bin_form = bin(val)[2:]
        if len(bin_form) > bit_size:
            # 如果二进制的位数，大于我们指定的位数，就报错
            raise SteganographyException("binary value larger than the expected size")
        while len(bin_form) < bit_size:
            # 前面 用0补足16位
            bin_form = "0" + bin_form
        return bin_form


    def byteValue(self, val):
        return self.binary_value(val, 8)


    def embed_text(self,txt):
        length = len(txt)
        # 把文本的长度编码16位的二进制，文本允许大小可以达到65536 bytes
        bin_length = self.binary_value(length,16)
        # 将文本长度的信息放进图片中
        self.put_binary_value(bin_length)
        # 将文本信息放进图片中
        for char in txt:
            # ord函数
            # 返回单个字符对应的 ASCII 数值，或者 Unicode 数值
            c = ord(char)
            # 将单个字符的 ASCII数值编码为8位二进制数，放入图像中
            self.put_binary_value(self.byteValue(c))

        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
        return self.image


    def encode_image(self, wm):
        w = wm.width
        h = wm.height
        if self.width * self.height * self.nbchannels < w * h * wm.channels:
            raise SteganographyException("要写入的水印图像超过原图大小！")
        binw = self.binary_value(w, 16)
        binh = self.binary_value(h, 16)
        self.put_binary_value(binw)
        self.put_binary_value(binh)
        for h in range(wm.height):
            for w in range(wm.width):
                for c in range(wm.channels):
                    val = wm[h, w][c]
                    self.put_binary_value(self.byteValue(int(val)))
        return self.image




if __name__ == '__main__':
    original_img = cv.imread('xiaozhi.jpg')
    lsb = LsbWatermarkEmbeder(original_img)
    processed_img = lsb.embed_text('FADADA')
    # cv.imwrite('xiaozhi_LSB.png',processed_img)

    plt.imshow(processed_img)
    plt.show()


    # processed = cv.imread('xiaozhi_LSB.png')
    # lsb = LSBWatermark(processed)
    # content = lsb.decode_text()
    # print(content)