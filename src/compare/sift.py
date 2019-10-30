import cv2 as cv
import numpy as np
import math
import time
from fddeye.compare import kd_tree
from fddeye.tools import logger
from fddeye.config import *


class Sift(object):
    '''sift 特征提取算法'''

    def __init__(self, sigma=SIGMA, bias=BIAS, intervals=INTERVALS,
                 max_interpolation_steps=MAX_INTERPOLATION_STEPS,
                 dxthreshold=DXTHRESHOLD, img_border=IMG_BORDER, ratio=RATIO,
                 ori_peak_ratio=ORI_PEAK_RATIO, descr_mag_thr=DESCR_MAG_THR,
                 archive=False):
        self.sigma = sigma
        self.bias = bias
        self.intervals = intervals
        self.max_interpolation_steps = max_interpolation_steps
        self.dxthreshold = dxthreshold
        self.img_border = img_border
        self.ratio = ratio
        self.ori_peak_ratio = ori_peak_ratio
        self.descr_mag_thr = descr_mag_thr
        self.archive = archive

    def is_grayscale(self, img):
        return len(img.shape) < 3

    def create_init_smooth_gray(self, img, sigma):
        """
        创建初始灰度图像
        初始图像先将原图像灰度化，再扩大一倍后，使用高斯模糊平滑
        :return:
        """
        if not self.is_grayscale(img):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, (0, 0), sigma, sigma)
        return img

    def gaussian_pyramid(self, img, octaves, intervals, sigma, archive=False):
        """
        高斯金字塔
        :return:
        """
        k = math.pow(2.0, 1.0 / intervals)
        invl = intervals + 3
        sigmas = np.ones((invl,))
        sigmas[0] = sigma
        for i in range(1, invl, 1):
            sig_prev = math.pow(k, i - 1) * sigma
            sig_total = sig_prev * k
            sigmas[i] = math.sqrt(sig_total * sig_total - sig_prev * sig_prev)

        gauss_pyr = [None] * octaves

        # for i in range(sigmas.shape[0]):
        #     logger.debug(sigmas[i])

        rows, cols = img.shape
        for o in range(octaves):
            gauss_pyr[o] = np.ones((invl, rows, cols))
            for i in range(invl):
                if o == 0 and i == 0:
                    gauss_pyr[o][i] = np.copy(img)
                elif i == 0:
                    # 前一组高斯图像的倒数第三层
                    # 第一组第一张图(下标为6)的图像是0组下标为3的图像降采样得来
                    # gauss_pyr[o][i] = down_sample(gauss_pyr[o-1][intervals])
                    gauss_pyr[o][i] = cv.pyrDown(gauss_pyr[o - 1][intervals], dstsize=(cols, rows))
                else:
                    gauss_pyr[o][i] = cv.GaussianBlur(gauss_pyr[o][i - 1], (0, 0), sigmas[i], sigmas[i])
            rows = rows // 2
            cols = cols // 2

        if archive:
            for o in range(octaves):
                for i in range(invl):
                    logger.debug(gauss_pyr[o][i].shape)
                    cv.imwrite('sample/gauss_{}_{}.png'.format(o, i), gauss_pyr[o][i],
                               [int(cv.IMWRITE_PNG_COMPRESSION), 3])

        return gauss_pyr

    def down_sample(self, src):
        rows, cols = src.shape
        rows2, cols2 = rows // 2, cols // 2
        dst = np.zeros((rows2, cols2))
        for i in range(rows2):
            for j in range(cols2):
                if 2 * i < rows and 2 * j < cols:
                    dst[i, j] = src[2 * i, 2 * j]
        return dst

    def dog_pyramid(self, gauss_pyr, octaves, intervals, archive=False):
        """
        差分金字塔
        :return:
        """
        dog_pyr = [None] * octaves
        invl = intervals + 3

        for o in range(octaves):
            dog_pyr[o] = np.ones((invl - 1,) + gauss_pyr[o][0].shape)
            for i in range(1, invl):
                dog_pyr[o][i - 1] = gauss_pyr[o][i] - gauss_pyr[o][i - 1]

        if archive:
            for o in range(octaves):
                layer = dog_pyr[o].copy()
                for i in range(invl - 1):
                    cv.normalize(layer[i], layer[i], 0, 255, cv.NORM_MINMAX)
                    img = layer[i].copy().astype(np.uint8)
                    cv.imwrite('sample/dog_{}_{}.png'.format(o, i), img, [int(cv.IMWRITE_PNG_COMPRESSION), 3])

        return dog_pyr

    def is_extremum(self, dog_pyr, octave, interval, x, y):
        val = dog_pyr[octave][interval, x, y]
        offset = (-1, 0, 1)

        sign = 1 if val > 0 else -1

        for i in offset:
            for j in offset:
                for k in offset:
                    val2 = dog_pyr[octave][interval + i, x + j, y + k]
                    if (sign * (val2 - val) > 0):
                        return False
        return True

    def get_offset_x(self, dog_pyr, octave, interval, x, y):
        # x^ = -H^(-1) * dx; dx = (Dx, Dy, Ds)^T
        h = self.hessian_3d(dog_pyr, octave, interval, x, y)
        h_inv = self.inverse_3d(h)
        if h_inv is None:
            h_inv = np.zeros((3, 3))
        d = self.derivative_3d(dog_pyr, octave, interval, x, y)
        d = d.reshape((3,))
        offset_x = -1 * (h_inv @ d)
        offset_x = offset_x.reshape((3,))
        return offset_x

    def get_fabs_dx(self, dog_pyr, octave, interval, x, y, offset):
        # |D(x^)|=D + 0.5 * dx * offset_x; dx=(Dx, Dy, Ds)^T
        dx = self.derivative_3d(dog_pyr, octave, interval, x, y)
        layer = dog_pyr[octave]
        d = layer[interval, x, y]
        dd = 0.5 * np.matmul(dx, offset)
        return math.fabs(d + dd)

    def hessian_3d(self, dog_pyr, octave, index, x, y):
        layer = dog_pyr[octave]
        h = np.zeros((3, 3))
        h[0, 0] = layer[index, x + 1, y] + layer[index, x - 1, y] - 2 * layer[index, x, y]
        h[1, 1] = layer[index, x, y + 1] + layer[index, x, y - 1] - 2 * layer[index, x, y]
        h[2, 2] = layer[index + 1, x, y] + layer[index - 1, x, y] - 2 * layer[index, x, y]
        h[1, 0] = h[0, 1] = (layer[index, x + 1, y + 1] + layer[index, x - 1, y - 1]
                             - layer[index, x + 1, y - 1] - layer[index, x - 1, y + 1]) / 4.0
        h[2, 0] = h[0, 2] = (layer[index + 1, x + 1, y] + layer[index - 1, x - 1, y]
                             - layer[index - 1, x + 1, y] - layer[index + 1, x - 1, y]) / 4.0
        h[2, 1] = h[1, 2] = (layer[index + 1, x, y + 1] + layer[index - 1, x, y - 1]
                             - layer[index + 1, x, y - 1] - layer[index - 1, x, y + 1]) / 4.0
        return h

    def derivative_3d(self, dog_pyr, octave, index, x, y):
        layer = dog_pyr[octave]
        d = np.zeros((3,))
        d[0] = (layer[index, x + 1, y] - layer[index, x - 1, y]) / 2.0
        d[1] = (layer[index, x, y + 1] - layer[index, x, y - 1]) / 2.0
        d[2] = (layer[index + 1, x, y] - layer[index - 1, x, y]) / 2.0
        return d

    def inverse_3d(self, h):
        try:
            return np.linalg.inv(h)
        except:
            return None

    def interploation_extremum(self, dog_pyr, octave, interval, x, y):
        """
        修正极值点，删除不稳定点
        |D(x)| < 0.03 Lowe 2004
        :return:
        """
        # 计算x=(x,y,sigma)^T
        img = dog_pyr[octave][interval]
        rows, cols = img.shape
        i = 0
        while i < self.max_interpolation_steps:
            offset_x = self.get_offset_x(dog_pyr, octave, interval, x, y)
            if np.all(offset_x < 0.5):
                break
            x += int(round(offset_x[0]))
            y += int(round(offset_x[1]))
            interval += int(round(offset_x[2]))

            # 此处保证检测边时 x+1,y+1和x-1, y-1有效
            if interval < 1 or interval > self.intervals \
                    or x >= rows - 1 or x < 2 or y >= cols - 1 or y < 2:
                return None
            i += 1

        if i >= self.max_interpolation_steps:
            return None

        # |D(x^)| < 0.03取经验值, rejecting unstable extrema
        offset_y = self.get_fabs_dx(dog_pyr, octave, interval, x, y, offset_x)
        if offset_y < self.dxthreshold / self.intervals:
            return None

        return self.create_keypoint(dog_pyr, octave, interval, x, y, offset_x=offset_x)

    def create_keypoint(self, dog_pyr, octave, interval, x, y, offset_x=np.zeros((3,))):
        keypoint = {}
        h, w = dog_pyr[octave][interval].shape
        keypoint['w'] = w
        keypoint['h'] = h
        keypoint['octave'] = octave

        keypoint['x'] = x
        keypoint['y'] = y
        keypoint['interval'] = interval

        keypoint['offset_x'] = offset_x[0]
        keypoint['offset_y'] = offset_x[1]
        keypoint['offset_interval'] = offset_x[2]

        keypoint['dx'] = (x + offset_x[0]) * pow(2.0, octave)
        keypoint['dy'] = (y + offset_x[1]) * pow(2.0, octave)

        keypoint['val'] = dog_pyr[octave][interval, x, y]

        return keypoint

    def pass_edge_response(self, dog_pyr, octave, interval, x, y):
        """
        hessian矩阵，排除边缘点
        eliminating edge responses
        :return:
        """
        # hessian矩阵
        #	   _ 	   _
        #    | Dxx  Dxy |
        # H =|			|
        #	 |_Dxy  Dyy_|
        #
        r = self.ratio
        img = dog_pyr[octave][interval]
        dxx = img[x + 1, y] + img[x - 1, y] - 2 * img[x, y]
        dyy = img[x, y + 1] + img[x, y - 1] - 2 * img[x, y]
        dxy = (img[x + 1, y + 1] + img[x - 1, y - 1] - img[x - 1, y + 1] - img[x + 1, y - 1]) / 4.0
        tr_h = dxx + dyy
        det_h = dxx * dyy - dxy * dxy

        if det_h <= 0:
            return False

        if tr_h * tr_h / det_h < (r + 1) * (r + 1) / r:
            return True
        return False

    def stride(self, src, kernals, strides):
        shape = src.shape
        src_stride = src.strides
        tile_shape = []
        tile_stride = []
        for i, dim in enumerate(shape):
            step = self.get_stride_step(dim, kernals[i], strides[i])
            if step <= 0:
                raise ValueError("无效的参数width={}, kernal={}, stride={}".format(dim, kernals[i], strides[i]))
            stride = strides[i] * src_stride[i]
            tile_shape.append(step)
            tile_stride.append(stride)
        dst_shape = tuple(tile_shape) + tuple(kernals)
        dst_stride = tuple(tile_stride) + tuple(src_stride)
        # logger.debug("shape={}, stride={}".format(dst_shape, dst_stride))
        return np.lib.stride_tricks.as_strided(src, shape=dst_shape, strides=dst_stride)

    def get_stride_step(self, width, kernal, stride):
        return (width - kernal) // stride + 1

    def detection_local_extrema2(self, dog_pyr, octaves, intervals, extrema_img=None):
        img_border = self.img_border
        thresh = 0.5 * self.dxthreshold / intervals
        extrmums = {"unstable": [], "stable": [], "edge": []}
        t = time.time()
        for o in range(octaves):
            layer = dog_pyr[o]
            _, rows, cols = layer.shape
            squares = self.stride(layer, (3, 3, 3), (1, 1, 1))
            max_v = squares.max(axis=(3, 4, 5))
            max_v[max_v < thresh] = 0.0
            min_v = squares.min(axis=(3, 4, 5))
            min_v[min_v > -thresh] = 0.0

            idx_i, idx_x, idx_y = np.nonzero((layer[1:-1, 1:-1, 1:-1] == max_v)
                                             | (layer[1:-1, 1:-1, 1:-1] == min_v))

            for i, x, y in zip(idx_i, idx_x, idx_y):
                if x + 1 < img_border or x + 1 >= rows - img_border: continue
                if y + 1 < img_border or y + 1 >= cols - img_border: continue

                # 修正极值点，删除不稳定点
                extrmum = self.interploation_extremum(dog_pyr, o, i + 1, x + 1, y + 1)
                if extrmum:
                    # hessian矩阵，排除边缘点
                    if self.pass_edge_response(dog_pyr, o, i + 1, x + 1, y + 1):
                        extrmums['stable'].append(extrmum)
                        # logger.debug("stable({},{},{},{})".format(o, i, extrmum['dx'], extrmum['dy']))
                    else:
                        extrmums['edge'].append(extrmum)
                        # logger.debug("edge({},{},{},{})".format(o, i, extrmum['dx'], extrmum['dy']))
                else:
                    extrmum = self.create_keypoint(dog_pyr, o, i + 1, x + 1, y + 1)
                    extrmums['unstable'].append(extrmum)
                    # logger.debug("unstable({},{},{},{})".format(o, i, extrmum['dx'], extrmum['dy']))

        stables = extrmums['stable']
        logger.debug("size={} cost={}".format(len(stables), (time.time() - t)))
        if extrema_img:
            self.draw_key_points(extrema_img, stables, color=(0, 0, 255), adjust=1)
            # draw_key_points(img, extremas['unstable'], color=(0, 255, 0), adjust=1)
            # draw_key_points(img, extremas['edge'], color=(255, 0, 0), adjust=1)
            cv.imwrite('sample/extremas.png', extrema_img, [int(cv.IMWRITE_PNG_COMPRESSION), 3])
        return extrmums

    def detection_local_extrema(self, dog_pyr, octaves, intervals, extrema_img=None):
        """
        检测当地极值点
        :return:
        """
        img_border = self.img_border
        extrmums = {"unstable": [], "stable": [], "edge": []}
        thresh = 0.5 * self.dxthreshold / intervals
        t = time.time()
        for o in range(octaves):
            for i in range(1, (intervals + 2) - 1):  # 第一层和最后一层极值忽略
                img = dog_pyr[o][i]
                rows, cols = img.shape
                for x in range(img_border, rows - img_border):
                    for y in range(img_border, cols - img_border):
                        val = img[x, y]
                        if math.fabs(val) <= thresh:  # 排除阈值过小的点
                            continue
                        if not self.is_extremum(dog_pyr, o, i, x, y):
                            continue

                        # 修正极值点，删除不稳定点
                        extrmum = self.interploation_extremum(dog_pyr, o, i, x, y)
                        if extrmum:
                            # hessian矩阵，排除边缘点
                            if self.pass_edge_response(dog_pyr, o, i, x, y):
                                extrmums['stable'].append(extrmum)
                                # logger.debug("stable({},{},{},{})".format(o, i, extrmum['dx'], extrmum['dy']))
                            else:
                                extrmums['edge'].append(extrmum)
                                # logger.debug("edge({},{},{},{})".format(o, i, extrmum['dx'], extrmum['dy']))
                        else:
                            extrmum = self.create_keypoint(dog_pyr, o, i, x, y)
                            extrmums['unstable'].append(extrmum)
                            # logger.debug("unstable({},{},{},{})".format(o, i, extrmum['dx'], extrmum['dy']))

        stables = extrmums['stable']
        logger.debug("size={} cost={}".format(len(stables), (time.time() - t)))
        if extrema_img:
            self.draw_key_points(extrema_img, stables, color=(0, 0, 255), adjust=1)
            # draw_key_points(img, extremas['unstable'], color=(0, 255, 0), adjust=1)
            # draw_key_points(img, extremas['edge'], color=(255, 0, 0), adjust=1)
            cv.imwrite('sample/extremas.png', extrema_img, [int(cv.IMWRITE_PNG_COMPRESSION), 3])
        return extrmums

    def calculate_scale(self, extremas, sigma, intervals):
        """
        计算尺度
        :return:
        """
        for extrema in extremas:
            intvl = extrema['interval'] + extrema['offset_interval']
            # 空间尺度坐标
            extrema['scale'] = sigma * math.pow(2.0, extrema['octave'] + intvl / intervals)
            # 高斯金字塔组内各层尺度坐标，不同组的相同层的sigma值相同
            # 关键点所在组的组内尺度
            extrema['octave_scale'] = sigma * math.pow(2.0, intvl / intervals)

    def orientation_assignment(self, gauss_pyr, extremas):
        """
        关键点方向分配
        :return:
        """
        features = []
        for extrema in extremas:
            hist = self.calculate_orientation_histogram(
                gauss_pyr[extrema['octave']][extrema['interval']], extrema['x'], extrema['y'],
                ORI_HIST_BINS, round(ORI_WINDOW_RADIUS * extrema['octave_scale']),
                ORI_SIGMA_TIMES * extrema['octave_scale'])
            self.gauss_smooth_ori_hist(hist, ORI_HIST_BINS, ORI_SMOOTH_TIMES)
            highest_peak = self.dominant_direction(hist)
            self.calc_ori_features(extrema, features, hist, ORI_HIST_BINS, highest_peak * self.ori_peak_ratio)

        return features

    def calc_ori_features(self, keypoint, features, hist, n, mag_thr):
        for i in range(n):
            l = (n + i - 1) % n
            r = (i + 1) % n
            if hist[i] > hist[l] and hist[i] > hist[r] and hist[i] >= mag_thr:
                bin = i + 0.5 * (hist[l] - hist[r]) / (hist[l] - 2.0 * hist[i] + hist[r])  # 抛物插值
                bin = bin if bin >= 0 else bin + n
                bin = bin if bin < n else bin - n
                dst = self.copy_keypoint(keypoint)
                # dst['oci'] = ((2 * math.pi * bin) / n) - math.pi
                # 角度修正
                oci = self.angle_to_radian(bin, bins=n)
                dst['oci'] = self.fold_radian(oci)
                features.append(dst)
        return features

    def copy_keypoint(self, src):
        dst = {}
        dst['w'] = src['w']
        dst['h'] = src['h']
        dst['octave'] = src['octave']
        dst['dx'] = src['dx']
        dst['dy'] = src['dy']

        dst['x'] = src['x']
        dst['y'] = src['y']
        dst['interval'] = src['interval']

        dst['offset_x'] = src['offset_x']
        dst['offset_y'] = src['offset_y']
        dst['offset_interval'] = src['offset_interval']

        dst['val'] = src['val']
        dst['octave_scale'] = src['octave_scale']
        dst['scale'] = src['scale']
        return dst

    def dominant_direction(self, hist):
        return np.max(hist)

    def gauss_smooth_ori_hist(self, hist, n, count):
        for c in range(count):
            for i in range(n):
                hist[i] = 0.25 * hist[(n + i - 1) % n] + 0.5 * hist[i] + 0.25 * hist[(i + 1) % n]

    def calculate_orientation_histogram(self, gauss, x, y, bins, radius, sigma):
        pi2 = 2.0 * math.pi
        econs = -1.0 / (2.0 * sigma * sigma)
        hist = np.zeros((bins,))

        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                ret = self.calc_grad_mag_ori(gauss, x + i, y + j)
                if ret:
                    mag, oci = ret
                    weight = math.exp((i * i + j * j) * econs)
                    # 使用Pi-ori将ori转换到[0,2*PI]之间
                    # bin = int(round(bins * (math.pi - oci) / pi2))
                    # bin = bin if bin < bins else 0
                    # 角度修正
                    oci = self.unfold_radian(oci)
                    bin = int(round(self.radian_to_angle(oci, bins=bins)))
                    bin = bin if bin < bins else 0
                    hist[bin] += mag * weight

        return hist

    def radian_to_angle(self, radian, bins=360):
        angle = bins * radian / (2 * math.pi)
        return angle

    def angle_to_radian(self, angle, bins=360):
        pi2 = 2 * math.pi
        radian = pi2 * angle / bins
        return radian

    def fold_radian(self, oci):
        '''
        将弧度由[0, 2*pi]变换到[-pi, pi]
        '''
        pi2 = 2 * math.pi
        while oci < 0:
            oci = oci + pi2
        while oci > pi2:
            oci = oci - pi2
        oci = oci if oci <= math.pi else oci - 2 * math.pi
        return oci

    def unfold_radian(self, oci):
        '''
        将弧度由[-pi, pi]变换到[0, 2*pi]
        '''
        pi2 = 2 * math.pi
        while oci < 0:
            oci = oci + pi2
        while oci > pi2:
            oci = oci - pi2
        return oci

    def calc_grad_mag_ori(self, gauss, x, y):
        rows, cols = gauss.shape
        if x > 0 and x < rows - 1 and y > 0 and y < cols - 1:
            dx = gauss[x + 1, y] - gauss[x - 1, y]
            dy = gauss[x, y + 1] - gauss[x, y - 1]

            mag = math.sqrt(dx * dx + dy * dy)
            # atan2返回[-Pi, Pi]的弧度值
            ori = math.atan2(dy, dx)
            return (mag, ori)
        else:
            return None

    def descriptor_representation(self, gauss_pyr, features, bins, width):
        """
        计算描述符
        :return:
        """
        i = 0
        for feature in features:
            hist = self.calculate_descr_hist(gauss_pyr[feature['octave']][feature['interval']],
                                             feature['x'], feature['y'], feature['octave_scale'], feature['oci'],
                                             bins, width)
            self.hist_to_descriptor(feature, hist)
            i += 1
        features.sort(key=self.feature_cmp)

    def feature_cmp(self, feature):
        return feature['scale']

    def hist_to_descriptor(self, feature, hist):
        hist = self.normalize_descr(hist)
        hist[hist > self.descr_mag_thr] = self.descr_mag_thr
        hist = self.normalize_descr(hist)
        hist = np.clip(INT_DESCR_FCTR * hist, 0, 255).astype(np.uint32)
        feature['descriptor'] = hist.ravel().tolist()
        feature['descr_length'] = hist.size

    def normalize_descr(self, arr):
        return arr / np.sqrt(np.sum(arr * arr))

    def calculate_descr_hist(self, gauss, x, y, octave_scale, ori, bins, width):
        hist = np.zeros((width, width, bins))
        # 角度修正
        cos_ori = math.cos(-ori)
        sin_ori = math.sin(-ori)
        # 高斯权值，sigma等于描述字窗口宽度的一半
        sigma = 0.5 * width
        conste = -1.0 / (2 * sigma * sigma)
        sub_hist_width = DESCR_SCALE_ADJUST * octave_scale
        pi2 = 2 * math.pi

        # 领域半径
        radius = int((sub_hist_width * math.sqrt(2.0) * (width + 1)) / 2.0 + 0.5)

        cost1 = 0
        cost2 = 0

        # 性能改进, 可以使用协程+进程
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                rot_x = (cos_ori * i - sin_ori * j) / sub_hist_width
                rot_y = (sin_ori * i + cos_ori * j) / sub_hist_width
                # xbin,ybin为落在4*4窗口中的下标值
                xbin = rot_x + width / 2 - 0.5
                ybin = rot_y + width / 2 - 0.5
                if xbin > -1.0 and xbin < width and ybin > -1.0 and ybin < width:
                    ret = self.calc_grad_mag_ori(gauss, x + i, y + j)
                    if ret:
                        grad_mag, grad_ori = ret
                        # grad_ori = (math.pi - grad_ori) - ori
                        # while grad_ori < 0.0:
                        #     grad_ori += pi2
                        # while grad_ori >= pi2:
                        #     grad_ori -= pi2
                        # obin = grad_ori * (bins / pi2)
                        # 角度修正
                        grad_ori = self.unfold_radian(grad_ori - ori)
                        obin = self.radian_to_angle(grad_ori, bins=bins)
                        weight = math.exp(conste * (rot_x * rot_x + rot_y * rot_y))
                        self.interp_hist_entry(hist, xbin, ybin, obin, grad_mag * weight, bins, width)

        return hist

    def interp_hist_entry(self, hist, xbin, ybin, obin, mag, bins, d):
        r0 = math.floor(xbin)
        c0 = math.floor(ybin)
        o0 = math.floor(obin)
        d_r = xbin - r0
        d_c = ybin - c0
        d_o = obin - o0
        '''
        做插值：
        xbin,ybin,obin:种子点所在子窗口的位置和方向
        所有种子点都将落在4*4的窗口中
        r0,c0取不大于xbin，ybin的正整数
        r0,c0只能取到0,1,2
        xbin,ybin在(-1, 2)

        r0取不大于xbin的正整数时。
        r0+0 <= xbin <= r0+1
        mag在区间[r0,r1]上做插值

        obin同理
        '''
        t = (0, 1)
        for r in t:
            rb = r0 + r
            if rb < 0 or rb >= d: continue
            v_r = mag * (1.0 - d_r if r == 0 else d_r)
            row = hist[rb]
            for c in t:
                cb = c0 + c
                if cb < 0 or cb >= d: continue
                v_c = v_r * (1.0 - d_c if c == 0 else d_c)
                h = row[cb]
                for o in t:
                    ob = (o0 + o) % bins
                    v_o = v_c * (1.0 - d_o if o == 0 else d_o)
                    h[ob] += v_o

    def extract(self, img, extrema_img=None):
        '''
        sift 算法
        :return:
        '''
        sigma = self.sigma
        intervals = self.intervals
        bias = self.bias
        # 创建初始灰度图像
        # 初始图像先将原图像灰度化，再扩大一倍后，使用高斯模糊平滑
        init_gray = self.create_init_smooth_gray(img, sigma)

        # 计算组数
        rows, cols = init_gray.shape
        octaves = math.floor(math.log2(min(rows, cols)) - bias)
        logger.debug(f"Target Info: size=({rows},{cols}), octaves={octaves}")

        # 高斯金字塔
        t = time.time()
        gauss_pyr = self.gaussian_pyramid(init_gray, octaves, intervals, sigma, archive=self.archive)
        logger.debug("生成高斯金字塔耗时: {}".format(time.time() - t))

        # 差分金字塔
        t = time.time()
        dog_pyr = self.dog_pyramid(gauss_pyr, octaves, intervals, archive=self.archive)
        logger.debug("生成差分金字塔耗时: {}".format(time.time() - t))

        # 检测当地极值点
        t = time.time()
        extremas = self.detection_local_extrema2(dog_pyr, octaves, intervals, extrema_img=extrema_img)
        stables = extremas['stable']
        logger.debug("检测当地极值点耗时: {}".format(time.time() - t))

        # 计算尺度
        self.calculate_scale(stables, sigma, intervals)

        # 关键点方向分配
        t = time.time()
        features = self.orientation_assignment(gauss_pyr, stables)
        logger.debug("关键点方向分配耗时: {}".format(time.time() - t))

        # 计算描述符
        t = time.time()
        self.descriptor_representation(gauss_pyr, features, DESCR_HIST_BINS, DESCR_WINDOW_WIDTH)
        logger.debug("计算描述符分配耗时: {}".format(time.time() - t))

        # 剔除无用信息
        w, h = img.shape[0:2]
        labels = self.get_labels(features, w, h, octaves, intervals, sigma)
        return labels

    def draw_key_points(self, img, keypoints, color=(0, 0, 255)):
        thickness = -1
        line_type = 8
        for keypoint in keypoints['labels']:
            center = (int(keypoint['y']), int(keypoint['x']))
            cv.circle(img, center, 3, color, thickness, line_type)
        return img

    def draw_sift_features(self, img, keypoints, color=(0, 255, 0)):
        for keypoint in keypoints['labels']:
            self.draw_sift_feature(img, keypoint, color)
        return img

    def draw_sift_feature(self, img, keypoint, color=(0, 255, 0)):
        scale = 5.0
        hscale = 0.75

        scl = keypoint['scale']
        ori = keypoint['theta']
        len = int(round(scl * scale))
        blen = len - int(round(scl * hscale))

        start_x = int(round(keypoint['y']))
        start_y = int(round(keypoint['x']))
        start = (start_x, start_y)
        end_x = int(round(len * math.cos(ori))) + start_x
        end_y = int(round(len * -math.sin(ori))) + start_y
        end = (end_x, end_y)
        h1_x = int(round(blen * math.cos(ori + math.pi / 18.0))) + start_x
        h1_y = int(round(blen * -math.sin(ori + math.pi / 18.0))) + start_y
        h1 = (h1_x, h1_y)
        h2_x = int(round(blen * math.cos(ori - math.pi / 18.0))) + start_x
        h2_y = int(round(blen * -math.sin(ori - math.pi / 18.0))) + start_y
        h2 = (h2_x, h2_y)

        cv.line(img, start, end, color, 1, 8, 0)
        cv.line(img, end, h1, color, 1, 8, 0)
        cv.line(img, end, h2, color, 1, 8, 0)
        return img

    def get_labels(self, features, w, h, octaves, intervals, sigma):
        datas = []
        labels = []
        metas = {'w': w, 'h': h, 'octaves': octaves, 'intervals': intervals, 'sigma': sigma}
        for feature in features:
            label = {}
            label['x'] = feature['dx']
            label['y'] = feature['dy']
            label['scale'] = feature['scale']
            label['theta'] = feature['oci']
            labels.append(label)
            datas.append(feature['descriptor'])
        return {"metas": metas, "datas": datas, "labels": labels}

    def write_features(self, features, file):
        pass

    def _match(self, ds1, lb1, ds2, lb2, threshold=NN_SQ_DIST_RATIO_THR, k=2):
        match = 0
        kd = kd_tree.KDTree(ds1, lb1)
        for d, lb in zip(ds2, lb2):
            nodes = kd.knn_algo(d, k=k)
            # logger.debug("nearest angle:", self.radian_to_angle(lb['theta'] - nodes[0][2]['theta']))
            if len(nodes) >= 2:
                if not math.isclose(nodes[1][0], 0.0, rel_tol=1e-9):
                    if nodes[0][0] / nodes[1][0] < threshold:
                        match += 1
            else:
                match += 1
        return match, len(ds2)

    def match(self, src, dst, threshold=NN_SQ_DIST_RATIO_THR, k=2):
        ds1, lb1 = src['datas'], src['labels']
        ds2, lb2 = dst['datas'], dst['labels']

        match1, size1 = self._match(ds1, lb1, ds2, lb2, threshold=threshold, k=k)
        return match1 / size1 if size1 != 0 else 0.0

    def bimatch(self, src, dst, threshold=NN_SQ_DIST_RATIO_THR, k=2):
        ds1, lb1 = src['datas'], src['labels']
        ds2, lb2 = dst['datas'], dst['labels']

        # match1, size1 = self._match(ds1, lb1, ds2, lb2, threshold=threshold, k=k)
        # match2, size2 = self._match(ds2, lb2, ds1, lb1, threshold=threshold, k=k)
        # logger.debug(match1, size1, match2, size2)
        # if size1 == 0 or size2 == 0:
        #     return 0.0
        # return self.harmonic_mean(match1 / size1, match2 / size2)
        match1, size1 = self._match(ds1, lb1, ds2, lb2, threshold=threshold, k=k)
        return match1 / size1

    def harmonic_mean(self, a, b):
        if a + b == 0:
            return 0.0
        return (2 * a * b) / (a + b)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Sift Extract Image Features')
    # parser.add_argument('--input', help='Path to input image.', default='person2.png')
    # args = parser.parse_args()

    # img1 = cv.imread("img/1.jpg", cv.IMREAD_COLOR)
    # features1 = sift(img1)
    #
    # draw_sift_features(img1, features1)
    # cv.imwrite('sample/features0.png', img1, [int(cv.IMWRITE_PNG_COMPRESSION), 3])
    #
    # datas, labels = get_labels(features1)
    # kd1 = kd_tree.KDTree(datas, labels)
    # logger.debug(kd1.length)

    img1 = cv.imread("img/1.jpg", cv.IMREAD_COLOR)
    sift = Sift()
    features1 = sift.extract(img1)

    sift.draw_sift_features(img1, features1)
    cv.imwrite('sample/features1.png', img1, [int(cv.IMWRITE_PNG_COMPRESSION), 3])

    img2 = cv.imread("img/2.jpg", cv.IMREAD_COLOR)
    features2 = sift.extract(img2)

    sift.draw_sift_features(img2, features2)
    cv.imwrite('sample/features2.png', img2, [int(cv.IMWRITE_PNG_COMPRESSION), 3])

    logger.debug("Match===================>")
    score1 = sift.match(features1, features2, k=2)
    score2 = sift.match(features2, features1, k=2)
    score3 = sift.bimatch(features1, features2, k=2)
    logger.debug("({}, {}, {})".format(score1, score2, score3))

    score1 = sift.match(features1, features2, k=5)
    score2 = sift.match(features2, features1, k=5)
    score3 = sift.bimatch(features1, features2, k=5)
    logger.debug("({}, {}, {})".format(score1, score2, score3))






