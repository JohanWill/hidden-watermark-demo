import hashlib
import textwrap
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont


np.set_printoptions(threshold=np.inf)

def generate_wm(name=None,icard=None):
    text = "张某某:350507188905349918"
    # text = ':'.join([name,icard])
    md5 = hashlib.md5()
    md5.update(text.encode('utf-8'))
    new_text = "张某某" + md5.hexdigest()
    lines = textwrap.wrap(new_text,width=10)

    background = np.zeros((128,200,3))
    background += 255
    background = Image.fromarray(np.uint8(background))
    font = ImageFont.truetype('../font/msyhbd.ttc', size=24)
    fillColor = (0,0,0)
    h = 5
    # if not isinstance(chinese, unicode):
    #     chinese = chinese.decode('utf-8')
    draw = ImageDraw.Draw(background) # 画笔句柄
    for line in lines:
        draw.text((10,h), line, font=font, fill=fillColor)
        h += 30

    background = cv.cvtColor(np.asarray(background), cv.COLOR_RGB2BGR)
    # cv.imshow('',background)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    cv.imwrite('../images/logo.png',background)
    return background

# # 第一个是x轴坐标，第二个是y轴坐标，原点在图片左下角
# point = (int(1500/15),int(200/2))
# background = cv.putText(background,md5.hexdigest(),point,fontFace=cv.FONT_HERSHEY_DUPLEX,fontScale=3,color=(0,0,0),thickness=8)

# plt.imshow(background)
# plt.show()


if __name__ == '__main__':
    background = generate_wm()
    # wm = cv.imread('../images/wm.png')
    # print(background)
    # print('---------------------------------------------------')
    # print(wm)