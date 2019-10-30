import cv2 as cv
import numpy as np
import json
import random
import math
import scipy.fftpack
from fddeye.compare import sift
from fddeye.tools import logger

np.seterr(over='ignore')

def show(name, img):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plt_show(layout, titles, images, cmap="binary"):
    import matplotlib.pyplot as plt
    layout = tuple(layout)
    image_size = 1
    for i in layout:
        image_size = image_size * i

    for i in range(image_size):
        args = layout + (i+1,)
        plt.subplot(*args)
        plt.title(titles[i])
        plt.imshow(images[i], cmap=cmap)
        plt.axis("off")

    plt.show()


def resize(img, size=300):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    img2 = cv.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    return img2

def rotate(img, angle, scale=1):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    dst = cv.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    return dst

def cosins(s1, s2):
    return np.sum(s1 * s2)/(np.linalg.norm(s1, 2) * np.linalg.norm(s2, 2))

def drawHist(hist, histSize, color=(255, 0, 0)):
    hist = np.array(hist)
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / histSize))
    histImage = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
    cv.normalize(hist, hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
    for i in range(1, histSize):
        cv.line(histImage, (bin_w * (i - 1), hist_h - int(round(hist[i - 1]))),
                (bin_w * (i), hist_h - int(round(hist[i]))),
                color, thickness=2)
    return histImage

def draw_arrow_img(size, arrow=0.5, color=(0, 0, 255)):
    size = size if size % 2 != 0 else size + 1
    arrow_size = int(arrow * (size - 1) / 2)
    if size < 20:
        raise ValueError("too small image size, size={}".format(size))

    img = np.zeros((size, size, 3))
    r = (size - 1) // 2
    line_type = 8
    # 中心圆
    cv.circle(img, (r, r), 3, color, -1, line_type)
    # 箭头
    thickness = 1
    end = (r + arrow_size, r)
    cv.line(img, (r, r), end, color, thickness, line_type)
    w, h = 7, 3
    r + arrow_size - w
    cv.line(img, (r + arrow_size - w, r - h), end, color, thickness, line_type)
    cv.line(img, (r + arrow_size - w, r + h), end, color, thickness, line_type)
    return img

def rotate_inside(src, angle, anchor=None, area=None,
                  src_mask=None, dst_mask=None, sweep=False):
    rows, cols = src.shape[:2]
    if src.ndim >= 3:
        channels = src.shape[2:]
    else:
        channels = tuple()
    if anchor is None:
        anchor = (rows // 2, cols // 2)
    if area is None:
        w = min(rows, cols)
        area = (w, w)
    h, w = area[0] // 2, area[1] // 2
    cx, cy = anchor
    r1, r2 = cx - h, cy + h
    c1, c2 = cx - w, cy + w
    r1 = r1 if r1 >= 0 else 0
    r2 = r2 if r2 < rows else rows - 1
    c1 = c1 if c1 >= 0 else 0
    c2 = c2 if c2 < cols else cols - 1

    radius = int(math.ceil(math.sqrt(h**2 + w**2)))
    canvas = np.zeros((radius * 2 + 1, radius * 2 + 1) + channels)
    canvas_center = (radius, radius)
    radian = angle_to_radian(angle)
    cos_ori = math.cos(radian)
    sin_ori = math.sin(radian)

    for i in range(r1, r2 + 1):
        for j in range(c1, c2 + 1):
            x, y = i - cx, j - cy
            val = src[i, j]
            if src_mask is None or src_mask(x, y):
                rot_x = (cos_ori * x - sin_ori * y)
                rot_y = (sin_ori * x + cos_ori * y)
                if dst_mask is None or dst_mask(rot_x, rot_y):
                    bilinear_interpolation(canvas, canvas_center, rot_x, rot_y, val)

    canvas.astype(src.dtype)
    cut_r = get_cut_size(rows, cols, cx, cy, radius)
    canvas = cut(canvas, (cut_r * 2 + 1, cut_r * 2 + 1), copy=(cut_r != radius))

    if sweep:
        src[r1:r2 + 1, c1:c2 + 1] = 0
    idx_x, idx_y = np.nonzero(np.any(canvas, axis=2))
    idx = idx_x - cut_r + cx, idx_y - cut_r + cy
    src[idx] = canvas[idx_x, idx_y]

def get_cut_size(rows, cols, cx, cy, r):
    min_x = min(cx, rows - 1 - cx)
    min_y = min(cy, cols - 1 - cy)
    min_r = min(r, min_x, min_y)
    return min_r


def bilinear_interpolation(canvas, anchor, rot_x, rot_y, val):
    cx, cy = anchor
    x = math.floor(rot_x)
    y = math.floor(rot_y)
    dx = rot_x - x
    dy = rot_y - y
    for i in (0, 1):
        for j in (0, 1):
            canvas[cx + x + i, cy + y + j] += val * (1 - dx if i == 0 else dx) * (1 - dy if j == 0 else dy)

def radian_to_angle(radian, bins=360):
    angle = bins * radian / (2 * math.pi)
    return angle

def angle_to_radian(angle, bins=360):
    pi2 = 2 * math.pi
    radian = pi2 * angle / bins
    return radian

def fold_radian(oci):
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

def unfold_radian(oci):
    '''
    将弧度由[-pi, pi]变换到[0, 2*pi]
    '''
    pi2 = 2 * math.pi
    while oci < 0:
        oci = oci + pi2
    while oci > pi2:
        oci = oci - pi2
    return oci


def cut(src, size=None, anchor=None, copy=False):
    rows, cols = src.shape[:2]
    if size is None:
        w = min(rows, cols)
        size = (w, w)
    if anchor is None:
        anchor = (rows // 2, cols // 2)
    h, w = size[0] //2, size[1] // 2
    r1, r2 = anchor[0] - h, anchor[0] + h
    c1, c2 = anchor[1] - w, anchor[1] + w
    r1 = r1 if r1 >= 0 else 0
    r2 = r2 if r2 < rows else rows - 1
    c1 = c1 if c1 >= 0 else 0
    c2 = c2 if c2 < cols else cols - 1
    dst = src[r1:r2 + 1, c1:c2 + 1]
    if copy:
        dst = dst.copy()
    return dst

class Compartor(object):
    def __init__(self):
        self.sift = sift.Sift(sigma=1.25, bias=3, dxthreshold=5, max_interpolation_steps=3, archive=False)
        self.features = {"sift", "dhash",  "hist"}
        self.sift_weights = {"sift": 0.7, "hist":0.3}
        self.weights = {"dhash": 0.7, "hist":0.3}

    def extract(self, img):
        features = {}
        for alg in self.features:
            method = getattr(self, "extract_" + alg)
            features[alg] = method(img)
        return json.dumps(features, indent=None)

    def get_sift_score(self, score):
        x1, x2 = (0.2, 0.6)
        y1, y2 = (0.5, 0.9)
        ret = y1 + (score - x1) * (y2 - y1) / (x2 - x1)
        ret = ret if ret <= y2 else 1
        ret = ret if ret >= y1 else 0
        return ret

    def compare(self, features1, features2):
        features1 = json.loads(features1)
        features2 = json.loads(features2)
        sum = 0.0

        if "sift" in self.features:
            score = self.compare_sift(features1, features2)
            if score > 0.2: # 相近
                for alg, weight in self.sift_weights.items():
                    if alg == 'sift':
                        sum += self.get_sift_score(score) * weight
                        continue
                    method = getattr(self, "compare_" + alg)
                    scores = method(features1[alg], features2[alg])
                    logger.debug("{} provide score is {}".format(alg, scores))
                    if isinstance(scores, tuple):
                        sum += self.multiscore(scores, weight)
                    else:
                        sum += scores * weight
                return sum

        for alg, weight in self.weights.items():
            method = getattr(self, "compare_" + alg)
            scores = method(features1[alg], features2[alg])
            logger.debug("{} provide score is {}".format(alg, scores))
            if isinstance(scores, tuple):
                sum += self.multiscore(scores, weight)
            else:
                sum += scores * weight
        return sum

    def multiscore(self, scores, weights):
        sum = 0.0
        for score, weight in zip(scores, weights):
            sum += score * weight
        return sum


    def extract_sift(self, img):
        img2 = resize(img, size=300)
        features = self.sift.extract(img2)
        tmp = self.sift.draw_sift_features(img2, features)
        r = random.randint(1,10)
        cv.imwrite('sample/features{}.png'.format(r), tmp, [int(cv.IMWRITE_PNG_COMPRESSION), 3])
        return features

    def compare_sift(self, features1, features2):
        score = self.sift.bimatch(features1, features2, threshold=0.8, k=5)
        if score > 0.5:
            return 0.8 + (score - 0.5) * 0.4
        else:
            return score

    def extract_hsv(self, img):
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        hsv_planes = cv.split(hsv_img)
        accumulate = False

        h_hist = cv.calcHist(hsv_planes, [0], None, [180], (0, 180), accumulate=accumulate)
        h_hist[:2] = 0; h_hist[-2:] = 0
        cv.normalize(h_hist, h_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        s_hist = cv.calcHist(hsv_planes, [1], None, [256], (0, 256), accumulate=accumulate)
        s_hist[:3] = 0; s_hist[-3:] = 0
        cv.normalize(s_hist, s_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        v_hist = cv.calcHist(hsv_planes, [2], None, [256], (0, 256), accumulate=accumulate)
        v_hist[:3] = 0; v_hist[-3:] = 0
        cv.normalize(v_hist, v_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        return {"h": h_hist.ravel().tolist(), "s": s_hist.ravel().tolist(), "v": v_hist.ravel().tolist()}

    def compare_hsv(self, hsv1, hsv2):
        h_score = cv.compareHist(np.array(hsv1['h'], dtype=np.float32), np.array(hsv2['h'], dtype=np.float32), cv.HISTCMP_CORREL)
        s_score = cv.compareHist(np.array(hsv1['s'], dtype=np.float32), np.array(hsv2['s'], dtype=np.float32), cv.HISTCMP_CORREL)
        v_score = cv.compareHist(np.array(hsv1['v'], dtype=np.float32), np.array(hsv2['v'], dtype=np.float32), cv.HISTCMP_CORREL)
        return abs(h_score), abs(s_score), abs(v_score)

    def extract_hist(self, img):
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        accumulate = False

        h_hist = cv.calcHist([hsv_img], [0], None, [256], (0, 256), accumulate=accumulate)
        h_hist[:3] = 0; h_hist[-3:] = 0
        cv.normalize(h_hist, h_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        return h_hist.ravel().tolist()

    def compare_hist(self, hist1, hist2):
        h_score = cv.compareHist(np.array(hist1, dtype=np.float32), np.array(hist2, dtype=np.float32), cv.HISTCMP_CORREL)
        return abs(h_score)

    def extract_grad_hist(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = resize(img, 300)
        bins = 180
        hist = np.zeros((bins,))

        h, w = img.shape[:2]
        for x in range(1,h-1):
            for y in range(1,w-1):
                ret = self.calc_grad_mag_ori(img, x, y)
                if ret:
                    mag, oci = ret
                    # 使用Pi-ori将ori转换到[0,2*PI]之间
                    # bin = int(round(bins * (math.pi - oci) / pi2))
                    # bin = bin if bin < bins else 0
                    # 角度修正
                    oci = unfold_radian(oci)
                    bin = int(round(radian_to_angle(oci, bins=bins)))
                    bin = bin if bin < bins else 0
                    hist[bin] += mag

        return hist.ravel().tolist()


    def compare_grad_hist(self, hist1, hist2):
        h_score = cv.compareHist(np.array(hist1, dtype=np.float32), np.array(hist2, dtype=np.float32), cv.HISTCMP_CORREL)
        return abs(h_score)

    def calc_grad_mag_ori(self, gauss, x, y):
        dx = gauss[x+1, y] - gauss[x-1, y]
        dy = gauss[x, y+1] - gauss[x, y-1]

        mag = math.sqrt(dx*dx + dy*dy)
        ori = math.atan2(dy, dx)
        return (mag, ori)

    def extract_contours(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_blur = cv.blur(gray_img, (3, 3))
        low_threshold = 20
        ratio = 3
        kernel_size = 3
        detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
        mask = detected_edges != 0
        dst = gray_img * (mask[:, :].astype(img.dtype))
        dst = self.shrink(dst, step=3, size=32, morph=cv.MORPH_DILATE, interpolation=cv.INTER_AREA)
        return dst.ravel().tolist()

    def compare_contours(self, contour1, contour2):
        score = cv.compareHist(np.array(contour1, dtype=np.float32), np.array(contour2, dtype=np.float32),
                               cv.HISTCMP_CORREL)
        return abs(score)

    def shrink(self, img, step=2, size=64, morph=cv.MORPH_DILATE, interpolation=cv.INTER_AREA):
        dilatation_size = 1
        element = cv.getStructuringElement(cv.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1), (dilatation_size, dilatation_size))
        h, w = img.shape
        while h / step >= size and w / step >= size:
            h, w = int(h / step), int(w / step)
            img = cv.morphologyEx(img, morph, element)
            img = cv.resize(img, (h, w), interpolation=interpolation)

        img = cv.resize(img, (size, size), interpolation=interpolation)
        return img

    def extract_phash(self, img):
        # 加载并调整图片为32x32灰度图片
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_CUBIC)

        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        vis0[:h, :w] = img  # 填充数据

        # 二维Dct变换
        # vis1 = cv.dct(cv.dct(vis0))
        vis1 = scipy.fftpack.dct(scipy.fftpack.dct(vis0, axis=0), axis=1)
        # cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
        vis1.resize(32, 32, refcheck=False)

        # 把二维list变成一维list
        img_list = vis1.flatten()

        # 计算均值
        avg = sum(img_list) * 1. / len(img_list)
        avg_list = ['0' if i > avg else '1' for i in img_list]

        # 得到哈希值
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])

    def compare_phash(self, hash1, hash2):
        return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(hash1, hash2)]) * 1. / (32 * 32 / 4)

    def extract_ahash(self, img):
        # 缩放为8*8
        img = cv.resize(img, (8, 8), interpolation=cv.INTER_CUBIC)
        # 转换为灰度图
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # s为像素和初值为0，hash_str为hash值初值为''
        hash_str = ''
        # 遍历累加求像素和
        avg = gray.mean()
        # 灰度大于平均值为1相反为0生成图片的hash值
        for i in range(8):
            for j in range(8):
                if gray[i, j] > avg:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    def comare_ahash(self, hash1, hash2):
        return self.cmpHash(hash1, hash2)

    def cmpHash(self, hash1, hash2):
        n = 0
        # hash长度不同则返回-1代表传参出错
        if len(hash1) != len(hash2):
            return -1
        # 遍历判断
        for i in range(len(hash1)):
            # 不相等则n计数+1，n最终为相似度
            if hash1[i] != hash2[i]:
                n = n + 1
        return 1 - n / 64

    def extract_dhash(self, img):
        # 缩放8*8
        img = cv.resize(img, (9, 8), interpolation=cv.INTER_CUBIC)
        # 转换灰度图
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hash_str = ''
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(8):
            for j in range(8):
                if gray[i, j] > gray[i, j + 1]:
                    hash_str = hash_str + '1'
                else:
                    hash_str = hash_str + '0'
        return hash_str

    def compare_dhash(self, hash1, hash2):
        return self.cmpHash(hash1, hash2)

    def extract_var(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        mean1, var1 = self.mean_vars(gray_img, size=64, axis=1)
        mean2, var2 = self.mean_vars(gray_img, size=64, axis=0)
        means, vars =  np.sqrt(mean1 * mean2), np.sqrt(var1 * var2)
        return (means.ravel().tolist(), vars.ravel().tolist())

    def compare_var(self, val1, val2):
        means1, vars1 = val1
        means2, vars2 = val2

        mscore = cv.compareHist(np.array(means1, dtype=np.float32), np.array(means2, dtype=np.float32), cv.HISTCMP_CORREL)
        vscore = cv.compareHist(np.array(vars1, dtype=np.float32), np.array(vars2, dtype=np.float32), cv.HISTCMP_CORREL)
        return min(abs(mscore), abs(vscore))

    def harmonic_mean(self, a, b):
        return (2 * a * b) / (a + b)

    def mean_vars(self, gray_img, size=64, axis=1):
        means = np.mean(gray_img, axis=axis)
        arr = np.array_split(means, size)
        means = np.zeros((64,))
        for i, item in enumerate(arr):
            means[i] = item.mean()
        vars = np.var(gray_img, axis=axis)
        arr = np.array_split(vars, size)
        vars = np.zeros((64,))
        for i, item in enumerate(arr):
            vars[i] = item.mean()
        return means, vars

if __name__ == '__main__':
    img1 = cv.imread("img/1.jpg", cv.IMREAD_COLOR)
    img1 = cut(img1)
    img2 = cv.imread("img/6.jpg", cv.IMREAD_COLOR)
    # img2 = rotate(img1, angle=90)
    # img2 = cv.imread("yaoxx.jpg", cv.IMREAD_COLOR)
    # logger.debug(img2.shape)
    show("img2", img2)
    eyer = Compartor()

    # hist1 = eyer.extract_grad_hist(img1)
    # hist2 = eyer.extract_grad_hist(img2)
    #
    # hist1 = drawHist(hist1, 180)
    # hist2 = drawHist(hist2, 180)
    #
    # plt_show((1, 2), ["original", "rotate"], [hist1, hist2])


    features1 = eyer.extract(img1)
    features2 = eyer.extract(img2)

    score = eyer.compare(features1, features2)
    logger.debug("score=", score)

    # features1 = eyer.extract_sift(img1)
    # features2 = eyer.extract_sift(img2)
    # logger.debug("sift score=", eyer.compare_sift(features1, features2))
    # hsv1 = eyer.extract_hsv(img1)
    # hsv2 = eyer.extract_hsv(img2)
    # logger.debug("hsv score=", eyer.compare_hsv(hsv1, hsv2))
    # img1 = eyer.extract_contours(img1)
    # img2 = eyer.extract_contours(img2)
    # score = eyer.compare_contours(img1, img2)
    # logger.debug("contours score=", score)

    # show("img1", img1)
    # show("img2", img2)
    # show("img3", img3)

    # cv.imwrite('contour1.png', contour1, [int(cv.IMWRITE_PNG_COMPRESSION), 3])







