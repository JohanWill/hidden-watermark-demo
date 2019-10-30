import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class SteganographyException(Exception):
    pass

class LsbWatermarkExtractor():
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


    def read_bit(self):
        """读取图像中的单个bit
        bit意为“位”或“比特”，是计算机运算的基础，属于二进制的范畴；
        Byte意为“字节”，是计算机文件大小的基本计算单位；
        """
        val = self.image[self.current_height, self.current_width][self.current_channel]
        val = int(val) & self.maskONE
        self.next_slot()
        if val > 0:
            return "1"
        else:
            return "0"



    def read_bits(self, nb):  # Read the given number of bits
        bits = ""
        for i in range(nb):
            bits += self.read_bit()
        return bits


    def read_byte(self):
        # 读取8位二进制 -> 单个字节
        return self.read_bits(8)


    def decode_text(self):
        # 读取二进制表示的文本大小
        lengths = self.read_bits(16)
        # 2表示是一串二进制的字符串
        length_int = int(lengths, 2)
        i = 0
        result = ""
        while i < length_int:
            tmp_bit = self.read_byte()
            # chr函数把[0~255]内的某个整数作参数，返回一个对应的字符。
            result += chr(int(tmp_bit, 2))
            i += 1
        return result


    def decode_image(self):
        width = int(self.read_bits(16), 2)
        height = int(self.read_bits(16), 2)
        wm = np.zeros((width, height, 3), np.uint8)
        for h in range(height):
            for w in range(width):
                for chan in range(wm.channels):
                    val = list(wm[h, w])
                    val[chan] = int(self.read_byte(), 2)
                    wm[h, w] = tuple(val)
        return wm