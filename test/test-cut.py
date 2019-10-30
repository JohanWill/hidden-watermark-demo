import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 3024,4032
def restore_image(embed_img,original_h,original_w,color='bgr'):
    if color == 'bgr':
        embed_img_rgb = cv.cvtColor(embed_img,cv.COLOR_BGR2RGB)
    elif color == 'rgb':
        embed_img_rgb = embed_img

    embed_h, embed_w, embed_c = embed_img_rgb.shape
    
    h_times = int(original_h/embed_h)
    w_times = int(original_w/embed_w)
    
    img = Image.fromarray(embed_img.astype(np.uint8))
    canvas = Image.new('RGB', (original_w,original_h)) # 先设定x轴长度，再设定y轴长度
    for i in range(h_times): # 1
        for j in range(w_times): # 2
            # 贴的时候也是，先输入x轴偏移，再输入y轴偏移
            canvas.paste(img,(embed_w * j,embed_h * i))
    
    # 裁剪操作img.crop((left, top, right, bottom))
    cut = img.crop((0,0,original_w - w_times * embed_w, embed_h))
    for i in range(h_times):
        canvas.paste(cut,(w_times * embed_w, i * embed_h))
    
    cut = img.crop((0,0,embed_w, original_h - embed_h * h_times))
    for j in range(w_times):
        canvas.paste(cut,(j * embed_w, h_times * embed_h))
    
    cut = img.crop((0,0,original_w - w_times * embed_w , original_h - embed_h * h_times))
    canvas.paste(cut,(w_times * embed_w, h_times * embed_h))
    
    return cv.cvtColor(np.array(canvas),cv.COLOR_RGB2BGR)
    


# tmp_img[:embed_h, embed_w,1] = embed_img[:original_w-embed_w,:,1]
# tmp_img[:embed_h, embed_w,2] = embed_img[:original_w-embed_w,:,2]
