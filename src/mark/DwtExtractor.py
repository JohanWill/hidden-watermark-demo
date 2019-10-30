import os
import time
import cv2
import numpy as np
from pywt import dwt2
from fddeye.tools import logger
from scipy.fftpack import dctn
from PIL import Image

class DwtWatermarkExractor():
    def __init__(self, random_seed_wm, random_seed_dct, mod, mod2=None,
                 wm_shape=None, block_shape=(4, 4),color_mod='YUV', dwt_deep=1):
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
    # 测试还原被裁剪的图像
    def restore_image(embed_img, original_h, original_w, color='bgr'):
        if color == 'bgr':
            embed_img_rgb = cv2.cvtColor(embed_img, cv2.COLOR_BGR2RGB)
        elif color == 'rgb':
            embed_img_rgb = embed_img

        embed_h, embed_w, embed_c = embed_img_rgb.shape

        h_times = int(original_h / embed_h)
        w_times = int(original_w / embed_w)

        img = Image.fromarray(embed_img.astype(np.uint8))
        canvas = Image.new('RGB', (original_w, original_h))  # 先设定x轴长度，再设定y轴长度
        for i in range(h_times):  # 1
            for j in range(w_times):  # 2
                # 贴的时候也是，先输入x轴偏移，再输入y轴偏移
                canvas.paste(img, (embed_w * j, embed_h * i))

        # 裁剪操作img.crop((left, top, right, bottom))
        cut = img.crop((0, 0, original_w - w_times * embed_w, embed_h))
        for i in range(h_times):
            canvas.paste(cut, (w_times * embed_w, i * embed_h))

        cut = img.crop((0, 0, embed_w, original_h - embed_h * h_times))
        for j in range(w_times):
            canvas.paste(cut, (j * embed_w, h_times * embed_h))

        cut = img.crop((0, 0, original_w - w_times * embed_w, original_h - embed_h * h_times))
        canvas.paste(cut, (w_times * embed_w, h_times * embed_h))

        return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

    def init_block_add_index(self, img_shape):
        # 假设原图长宽均为2的整数倍,同时假设水印为64*64,则32*32*4
        # 分块并DCT
        shape0_int, shape1_int = int(img_shape[0] / self.block_shape[0]), int(img_shape[1] / self.block_shape[1])
        if not shape0_int * shape1_int >= self.wm_shape[0] * self.wm_shape[1]:
            raise Exception("水印的大小超过图片的容量,不是完整的水印")
        self.part_shape = (shape0_int * self.block_shape[0], shape1_int * (self.block_shape[1]))
        self.block_add_index0, self.block_add_index1 = np.meshgrid(np.arange(shape0_int), np.arange(shape1_int))
        self.block_add_index0, self.block_add_index1 = self.block_add_index0.flatten(), self.block_add_index1.flatten()
        self.length = self.block_add_index0.size
        assert self.block_add_index0.size == self.block_add_index1.size

    def block_get_wm(self, block, index):
        block_dct = dctn(block, norm='ortho')
        block_dct_flatten = block_dct.flatten().copy()
        block_dct_flatten = block_dct_flatten[index]
        block_dct_shuffled = block_dct_flatten.reshape(self.block_shape)

        U, s, V = np.linalg.svd(block_dct_shuffled)
        max_s = s[0]
        wm_1 = 255 if max_s % self.mod > self.mod / 2 else 0
        if self.mod2:
            max_s = s[1]
            wm_2 = 255 if max_s % self.mod2 > self.mod2 / 2 else 0
            wm = (wm_1 * 3 + wm_2 * 1) / 4
        else:
            wm = wm_1
        return wm

    def extract(self, filename,output_path=None):
        if not isinstance(filename,str):
            embed_img = filename.astype(np.float32)
        else:
            embed_img = cv2.imread(filename).astype(np.float32)
        if self.color_mod == 'RGB':
            embed_img_YUV = embed_img
        elif self.color_mod == 'YUV':
            embed_img_YUV = cv2.cvtColor(embed_img, cv2.COLOR_BGR2YUV)

        if not embed_img_YUV.shape[0] % (2 ** self.dwt_deep) == 0:
            temp = (2 ** self.dwt_deep) - embed_img_YUV.shape[0] % (2 ** self.dwt_deep)
            embed_img_YUV = np.concatenate((embed_img_YUV, np.zeros((temp, embed_img_YUV.shape[1], 3))), axis=0)
        if not embed_img_YUV.shape[1] % (2 ** self.dwt_deep) == 0:
            temp = (2 ** self.dwt_deep) - embed_img_YUV.shape[1] % (2 ** self.dwt_deep)
            embed_img_YUV = np.concatenate((embed_img_YUV, np.zeros((embed_img_YUV.shape[0], temp, 3))), axis=1)

        assert embed_img_YUV.shape[0] % (2 ** self.dwt_deep) == 0
        assert embed_img_YUV.shape[1] % (2 ** self.dwt_deep) == 0
        logger.info("读取图片成功！要抽取水印的图片大小为 ： {0}".format(embed_img_YUV.shape))
        logger.info("开始进行{}次小波变换".format(self.dwt_deep))
        embed_img_Y = embed_img_YUV[:, :, 0]
        embed_img_U = embed_img_YUV[:, :, 1]
        embed_img_V = embed_img_YUV[:, :, 2]
        coeffs_Y = dwt2(embed_img_Y, 'haar')
        coeffs_U = dwt2(embed_img_U, 'haar')
        coeffs_V = dwt2(embed_img_V, 'haar')
        ha_Y = coeffs_Y[0]
        ha_U = coeffs_U[0]
        ha_V = coeffs_V[0]
        # 对ha进一步进行小波变换,并把下一级ha保存到ha中
        for i in range(self.dwt_deep - 1):
            coeffs_Y = dwt2(ha_Y, 'haar')
            ha_Y = coeffs_Y[0]
            coeffs_U = dwt2(ha_U, 'haar')
            ha_U = coeffs_U[0]
            coeffs_V = dwt2(ha_V, 'haar')
            ha_V = coeffs_V[0]

        # 初始化块索引数组
        try:
            if self.ha_Y.shape == ha_Y.shape:
                self.init_block_add_index(ha_Y.shape)
            else:
                logger.warn('你现在要解水印的图片与之前读取的原图的形状不同,这是不被允许的')
        except:
            self.init_block_add_index(ha_Y.shape)

        ha_block_shape = (
        int(ha_Y.shape[0] / self.block_shape[0]), int(ha_Y.shape[1] / self.block_shape[1]), self.block_shape[0],
        self.block_shape[1])
        strides = ha_Y.itemsize * (
            np.array([ha_Y.shape[1] * self.block_shape[0], self.block_shape[1], ha_Y.shape[1], 1]))

        ha_Y_block = np.lib.stride_tricks.as_strided(ha_Y.copy(), ha_block_shape, strides)
        ha_U_block = np.lib.stride_tricks.as_strided(ha_U.copy(), ha_block_shape, strides)
        ha_V_block = np.lib.stride_tricks.as_strided(ha_V.copy(), ha_block_shape, strides)

        extract_wm = np.array([])
        extract_wm_Y = np.array([])
        extract_wm_U = np.array([])
        extract_wm_V = np.array([])
        self.random_dct = np.random.RandomState(self.random_seed_dct)

        start = time.time()
        logger.info("小波变换完成，开始提取水印")

        index = np.arange(self.block_shape[0] * self.block_shape[1])
        for i in range(self.length):
            if i % 5000 == 0:
                logger.debug(f"正在提取第{i+1}个窗口的像素点")
            self.random_dct.shuffle(index)
            wm_Y = self.block_get_wm(ha_Y_block[self.block_add_index0[i], self.block_add_index1[i]], index)
            wm_U = self.block_get_wm(ha_U_block[self.block_add_index0[i], self.block_add_index1[i]], index)
            wm_V = self.block_get_wm(ha_V_block[self.block_add_index0[i], self.block_add_index1[i]], index)
            wm = round((wm_Y + wm_U + wm_V) / 3)

            # else情况是对循环嵌入的水印的提取
            if i < self.wm_shape[0] * self.wm_shape[1]:
                extract_wm = np.append(extract_wm, wm)
                extract_wm_Y = np.append(extract_wm_Y, wm_Y)
                extract_wm_U = np.append(extract_wm_U, wm_U)
                extract_wm_V = np.append(extract_wm_V, wm_V)
            else:
                times = int(i / (self.wm_shape[0] * self.wm_shape[1]))
                ii = i % (self.wm_shape[0] * self.wm_shape[1])
                extract_wm[ii] = (extract_wm[ii] * times + wm) / (times + 1)
                extract_wm_Y[ii] = (extract_wm_Y[ii] * times + wm_Y) / (times + 1)
                extract_wm_U[ii] = (extract_wm_U[ii] * times + wm_U) / (times + 1)
                extract_wm_V[ii] = (extract_wm_V[ii] * times + wm_V) / (times + 1)

        wm_index = np.arange(extract_wm.size)
        self.random_wm = np.random.RandomState(self.random_seed_wm)
        self.random_wm.shuffle(wm_index)
        extract_wm[wm_index] = extract_wm.copy()
        extract_wm_Y[wm_index] = extract_wm_Y.copy()
        extract_wm_U[wm_index] = extract_wm_U.copy()
        extract_wm_V[wm_index] = extract_wm_V.copy()

        end = time.time()
        logger.info(f"水印提取完成，总共耗时 ： {end-start}")

        if output_path:
            cv2.imwrite(output_path,extract_wm.reshape(self.wm_shape[0], self.wm_shape[1]))
        return extract_wm.reshape(self.wm_shape[0], self.wm_shape[1])

        # path, file_name = os.path.split(out_wm_name)
        # cv2.imwrite(os.path.join(path, 'Y_U_V', 'Y' + file_name), extract_wm_Y.reshape(self.wm_shape[0], self.wm_shape[1]))
        # cv2.imwrite(os.path.join(path, 'Y_U_V', 'U' + file_name), extract_wm_U.reshape(self.wm_shape[0], self.wm_shape[1]))
        # cv2.imwrite(os.path.join(path, 'Y_U_V', 'V' + file_name), extract_wm_V.reshape(self.wm_shape[0], self.wm_shape[1]))

if __name__ == '__main__':
    extractor = DwtWatermarkExractor(4399,2333,32,wm_shape=(128,200),dwt_deep=2)
    if os.path.exists('../debug-output/lena-out.png'):
        os.remove('../debug-output/lena-out.png')
    extractor.extract('../debug-output/lena.png', '../debug-output/lena-out.png')
    # extractor.extract('../debug-output/debug-embed-watermark.png','../debug-output/debug-extract-watermark.png')