import os
import time
import hashlib
import textwrap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pywt import dwt2,idwt2
from fddeye.config import *
from fddeye.tools import logger
from scipy.fftpack import dctn,idctn
from PIL import Image,ImageDraw,ImageFont

class DWTWatermarkEmbeder():
    def __init__(self, random_seed_wm, random_seed_dct, mod, mod2=None,
                 wm_shape=None, block_shape=(4, 4), color_mod='YUV',dwt_deep=1):
        # self.wm_per_block = 1
        self.block_shape = block_shape  # 2^n
        self.random_seed_wm = random_seed_wm
        self.random_seed_dct = random_seed_dct
        self.mod = mod
        self.mod2 = mod2
        self.wm_shape = wm_shape
        self.color_mod = color_mod
        self.dwt_deep = dwt_deep

    @staticmethod
    def generate_wm(time_text):
        md5 = hashlib.md5()
        md5.update(time_text.encode('utf-8'))
        new_text = md5.hexdigest()
        lines = textwrap.wrap(new_text, width=WATER_MARK_TEXT_WARP_WIDTH)

        background = np.zeros((WATER_MARK_Y, WATER_MARK_X, 3))
        background += 255
        background = Image.fromarray(np.uint8(background))
        font = ImageFont.truetype(f'../font/{WATER_MARK_FONT}', size=WATER_MARK_FONT_SIZE)
        fillColor = (0, 0, 0)
        h = 5
        # if not isinstance(chinese, unicode):
        #     chinese = chinese.decode('utf-8')
        draw = ImageDraw.Draw(background)  # 画笔句柄
        for line in lines:
            draw.text((10, h), line, font=font, fill=fillColor)
            h += WATER_MARK_TEXT_LINE_SPACE

        background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
        return background,new_text

    def init_block_add_index(self, img_shape): # (135,240)
        shape0_int, shape1_int = int(img_shape[0] / self.block_shape[0]), int(img_shape[1] / self.block_shape[1])
        if not shape0_int * shape1_int >= self.wm_shape[0] * self.wm_shape[1]:
            raise Exception("水印的大小超过图片的容量,无法嵌入完整水印")
        self.part_shape = (shape0_int * self.block_shape[0], shape1_int * (self.block_shape[1]))
        self.block_add_index0, self.block_add_index1 = np.meshgrid(np.arange(shape0_int), np.arange(shape1_int))
        self.block_add_index0, self.block_add_index1 = self.block_add_index0.flatten(), self.block_add_index1.flatten()
        self.length = self.block_add_index0.size
        assert self.block_add_index0.size == self.block_add_index1.size


    def read_ori_img(self, filename):
        if isinstance(filename, str):
            ori_img = cv2.imread(filename).astype(np.float32)
        else:
            ori_img = filename.astype(np.float32)
        self.ori_img_shape = ori_img.shape[:2]
        if self.color_mod == 'RGB':
            self.ori_img_YUV = ori_img
        elif self.color_mod == 'YUV':
            self.ori_img_YUV = cv2.cvtColor(ori_img, cv2.COLOR_BGR2YUV)

        if not self.ori_img_YUV.shape[0] % (2 ** self.dwt_deep) == 0:
            temp = (2 ** self.dwt_deep) - self.ori_img_YUV.shape[0] % (2 ** self.dwt_deep)
            self.ori_img_YUV = np.concatenate((self.ori_img_YUV, np.zeros((temp, self.ori_img_YUV.shape[1], 3))), axis=0)
        if not self.ori_img_YUV.shape[1] % (2 ** self.dwt_deep) == 0:
            temp = (2 ** self.dwt_deep) - self.ori_img_YUV.shape[1] % (2 ** self.dwt_deep)
            self.ori_img_YUV = np.concatenate((self.ori_img_YUV, np.zeros((self.ori_img_YUV.shape[0], temp, 3))), axis=1)
        assert self.ori_img_YUV.shape[0] % (2 ** self.dwt_deep) == 0
        assert self.ori_img_YUV.shape[1] % (2 ** self.dwt_deep) == 0

        logger.info("开始对原图进行{0}次二维离散小波变换".format(self.dwt_deep))
        if self.dwt_deep == 1:
            coeffs_Y = dwt2(self.ori_img_YUV[:, :, 0], 'haar')
            ha_Y = coeffs_Y[0]
            coeffs_U = dwt2(self.ori_img_YUV[:, :, 1], 'haar')
            ha_U = coeffs_U[0]
            coeffs_V = dwt2(self.ori_img_YUV[:, :, 2], 'haar')
            ha_V = coeffs_V[0]
            self.coeffs_Y = [coeffs_Y[1]]
            self.coeffs_U = [coeffs_U[1]]
            self.coeffs_V = [coeffs_V[1]]

        elif self.dwt_deep >= 2:
            # 不希望使用太多级的dwt,2,3次就行了
            coeffs_Y = dwt2(self.ori_img_YUV[:, :, 0], 'haar')
            ha_Y = coeffs_Y[0]
            coeffs_U = dwt2(self.ori_img_YUV[:, :, 1], 'haar')
            ha_U = coeffs_U[0]
            coeffs_V = dwt2(self.ori_img_YUV[:, :, 2], 'haar')
            ha_V = coeffs_V[0]
            self.coeffs_Y = [coeffs_Y[1]]
            self.coeffs_U = [coeffs_U[1]]
            self.coeffs_V = [coeffs_V[1]]
            for i in range(self.dwt_deep - 1):
                coeffs_Y = dwt2(ha_Y, 'haar')
                ha_Y = coeffs_Y[0]
                coeffs_U = dwt2(ha_U, 'haar')
                ha_U = coeffs_U[0]
                coeffs_V = dwt2(ha_V, 'haar')
                ha_V = coeffs_V[0]
                self.coeffs_Y.append(coeffs_Y[1])
                self.coeffs_U.append(coeffs_U[1])
                self.coeffs_V.append(coeffs_V[1])
        self.ha_Y = ha_Y
        self.ha_U = ha_U
        self.ha_V = ha_V
        logger.info("完成二维离散小波变换,变换后的图片大小为{0}".format(self.ha_Y.shape))

        self.ha_block_shape = (
        int(self.ha_Y.shape[0] / self.block_shape[0]), int(self.ha_Y.shape[1] / self.block_shape[1]), self.block_shape[0],
        self.block_shape[1])
        strides = self.ha_Y.itemsize * (
            np.array([self.ha_Y.shape[1] * self.block_shape[0], self.block_shape[1], self.ha_Y.shape[1], 1]))

        self.ha_Y_block = np.lib.stride_tricks.as_strided(self.ha_Y.copy(), self.ha_block_shape, strides)
        self.ha_U_block = np.lib.stride_tricks.as_strided(self.ha_U.copy(), self.ha_block_shape, strides)
        self.ha_V_block = np.lib.stride_tricks.as_strided(self.ha_V.copy(), self.ha_block_shape, strides)


    def read_wm(self, filename):
        if isinstance(filename,str):
            self.wm = cv2.imread(filename)[:, :, 0]
        else:
            self.wm = filename[:, :, 0]
        self.wm_shape = self.wm.shape[:2]

        # 初始化块索引数组
        self.init_block_add_index(self.ha_Y.shape)

        self.wm_flatten = self.wm.flatten()
        if self.random_seed_wm:
            self.random_wm = np.random.RandomState(self.random_seed_wm)
            self.random_wm.shuffle(self.wm_flatten)


    def block_add_wm(self, block, index, i):
        i = i % (self.wm_shape[0] * self.wm_shape[1])

        wm_1 = self.wm_flatten[i]
        block_dct = dctn(block, norm='ortho')
        block_dct_flatten = block_dct.flatten().copy()

        block_dct_flatten = block_dct_flatten[index]
        block_dct_shuffled = block_dct_flatten.reshape(self.block_shape)
        U, s, V = np.linalg.svd(block_dct_shuffled)
        max_s = s[0]
        s[0] = (max_s - max_s % self.mod + 3 / 4 * self.mod) if wm_1 >= 128 else (
                    max_s - max_s % self.mod + 1 / 4 * self.mod)
        if self.mod2:
            max_s = s[1]
            s[1] = (max_s - max_s % self.mod2 + 3 / 4 * self.mod2) if wm_1 >= 128 else (
                        max_s - max_s % self.mod2 + 1 / 4 * self.mod2)
        # s[1] = (max_s-max_s%self.mod2+3/4*self.mod2) if wm_1<128 else (max_s-max_s%self.mod2+1/4*self.mod2)

        ###np.dot(U[:, :k], np.dot(np.diag(sigma[:k]),v[:k, :]))
        block_dct_shuffled = np.dot(U, np.dot(np.diag(s), V))

        block_dct_flatten = block_dct_shuffled.flatten()

        block_dct_flatten[index] = block_dct_flatten.copy()
        block_dct = block_dct_flatten.reshape(self.block_shape)

        return idctn(block_dct, norm='ortho')


    def embed(self,ori_img,wm_img,filename=None):
        self.read_ori_img(ori_img)
        self.read_wm(wm_img)

        embed_ha_Y_block = self.ha_Y_block.copy()
        embed_ha_U_block = self.ha_U_block.copy()
        embed_ha_V_block = self.ha_V_block.copy()

        self.random_dct = np.random.RandomState(self.random_seed_dct)
        index = np.arange(self.block_shape[0] * self.block_shape[1])

        start = time.time()
        logger.info("开始遍历坐标矩阵中的每个点，一共有{}个window".format(self.length))
        for i in range(self.length):
            self.random_dct.shuffle(index)
            embed_ha_Y_block[self.block_add_index0[i], self.block_add_index1[i]] = self.block_add_wm(
                embed_ha_Y_block[self.block_add_index0[i], self.block_add_index1[i]], index, i)
            embed_ha_U_block[self.block_add_index0[i], self.block_add_index1[i]] = self.block_add_wm(
                embed_ha_U_block[self.block_add_index0[i], self.block_add_index1[i]], index, i)
            embed_ha_V_block[self.block_add_index0[i], self.block_add_index1[i]] = self.block_add_wm(
                embed_ha_V_block[self.block_add_index0[i], self.block_add_index1[i]], index, i)
            if i % 5000 == 0:
                logger.debug(f"完成对第{(i+1)}个window,YUV三个分量的嵌入")
        end = time.time()
        logger.info(f"水印嵌入完成。总共耗时 ： {(end-start)}")

        start = time.time()
        logger.info("开始合并YUV通道,准备反向小波变换")
        embed_ha_Y_part = np.concatenate(embed_ha_Y_block, 1)
        embed_ha_Y_part = np.concatenate(embed_ha_Y_part, 1)
        embed_ha_U_part = np.concatenate(embed_ha_U_block, 1)
        embed_ha_U_part = np.concatenate(embed_ha_U_part, 1)
        embed_ha_V_part = np.concatenate(embed_ha_V_block, 1)
        embed_ha_V_part = np.concatenate(embed_ha_V_part, 1)

        embed_ha_Y = self.ha_Y.copy()
        embed_ha_Y[:self.part_shape[0], :self.part_shape[1]] = embed_ha_Y_part
        embed_ha_U = self.ha_U.copy()
        embed_ha_U[:self.part_shape[0], :self.part_shape[1]] = embed_ha_U_part
        embed_ha_V = self.ha_V.copy()
        embed_ha_V[:self.part_shape[0], :self.part_shape[1]] = embed_ha_V_part

        for i in range(self.dwt_deep):
            (cH, cV, cD) = self.coeffs_Y[-1 * (i + 1)]
            embed_ha_Y = idwt2((embed_ha_Y.copy(), (cH, cV, cD)), "haar")  # 其idwt得到父级的ha
            (cH, cV, cD) = self.coeffs_U[-1 * (i + 1)]
            embed_ha_U = idwt2((embed_ha_U.copy(), (cH, cV, cD)), "haar")  # 其idwt得到父级的ha
            (cH, cV, cD) = self.coeffs_V[-1 * (i + 1)]
            embed_ha_V = idwt2((embed_ha_V.copy(), (cH, cV, cD)), "haar")  # 其idwt得到父级的ha
            # 最上级的ha就是嵌入水印的图,即for运行完的ha

        embed_img_YUV = np.zeros(self.ori_img_YUV.shape, dtype=np.float32)
        embed_img_YUV[:, :, 0] = embed_ha_Y
        embed_img_YUV[:, :, 1] = embed_ha_U
        embed_img_YUV[:, :, 2] = embed_ha_V

        embed_img_YUV = embed_img_YUV[:self.ori_img_shape[0], :self.ori_img_shape[1]]
        if self.color_mod == 'RGB':
            embed_img = embed_img_YUV
        elif self.color_mod == 'YUV':
            embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        end = time.time()
        logger.info(f"反向小波变换完成,耗时 ： {(end-start)}")

        embed_img[embed_img > 255] = 255
        embed_img[embed_img < 0] = 0
        if filename:
            cv2.imwrite(filename, embed_img)
        return embed_img

if __name__ == '__main__':
    embeder = DWTWatermarkEmbeder(4399,2333,32,dwt_deep=2)
    if os.path.exists('../debug-output/lena.png'):
        os.remove('../debug-output/lena.png')
    # embeder.embed('../../images/lena.png','../../images/wm.png','../debug-output/lena.png')
    embeder.embed('../../images/xiaozhi.jpg', '../debug-output/debug-generate-watermark.png', '../debug-output/lena.png')