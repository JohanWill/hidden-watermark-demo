"""
一个嵌入和提取数字水印的项目，支持两张图片的多种算法比对。
测试用例：
python eye.py debug --embed --original_file_path ../images/test_cmyk.jpg --watermark_file_path ./debug-output/debug-generate-watermark.png
python eye.py debug -e --extract_file_path ./debug-output/debug-embed-watermark.png
python eye.py debug -g  --name 差不多 --id 312391432483842

编译window exe文件
$python setup.py bdist_wininst
"""
import os
import sys
import time
import arrow
import hashlib
import argparse
import cv2 as cv
import numpy as np
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
# To find local version of the library
sys.path.append(ROOT_DIR)
from fddeye.config import *
from fddeye.tools import logger
from fddeye.mysql_client import MysqlClient
from fddeye.compare.compartor import Compartor
from fddeye.mark.DwtEmbeder import DWTWatermarkEmbeder
from fddeye.mark.DwtExtractor import DwtWatermarkExractor
import fddeye.txcos_ops as cos



parser = argparse.ArgumentParser(description="Fadada Eye Program Contain Watermark Operation and  Image Contras")
parser.add_argument("mode",type=str,choices=['debug','run'],help="The program mode")
parser.add_argument("--name",type=str,help="name need to generate watermark")
parser.add_argument("--id",type=str,help="id number need to generate watermark")
parser.add_argument("--original_file_path",type=str,help="path to image need to embed watermark")
parser.add_argument("--watermark_file_path",type=str,help="path to watermark pic")
parser.add_argument("--extract_file_path",type=str,help="path to image need extract watermark ")
parser.add_argument("--output_path",type=str,help="if None,default is current image folder")

group = parser.add_mutually_exclusive_group() # 一个互斥参数组
group.add_argument('-g','--gw',help="generate a water mark",action="store_true")
group.add_argument('-d','--embed',help="embed a water mark into image",action="store_true")
group.add_argument('-e','--extract',help="extract a water mark from image",action="store_true")
args = parser.parse_args()


def query_img_to_wm(mysql_client, compartor, pool):
    count, result = mysql_client.select_many(
        f"""
        SELECT 
            imgNumber,
            originalPicX,
            originalPicY
        FROM 
            {DB_TABLE} 
        WHERE waterMark = 0 or waterMark is NULL 
        """)
    logger.debug(f"查询{DB_TABLE}中未添加水印的图片，总共找到{count}条记录")

    result_dict = {}
    for line in result:
        result_dict['imgNumber'] = line[0]
        result_dict['originalPicX'] = line[1]
        result_dict['originalPicY'] = line[2]
        pool.submit(add_watermark,result_dict.copy(),mysql_client, compartor).add_done_callback(upload_filepath_to_cos)
        logger.debug("当前线程池队列堆积长度：{}".format(getattr(pool, '_work_queue')._qsize()))


def add_watermark(task_dict,mysql_client, compartor):
    imgNumber = task_dict['imgNumber']
    response, image = cos.get_img_file(imgNumber)
    x = task_dict['originalPicX']
    y = task_dict['originalPicY']

    assert x >= WATER_MARK_X * EMBEDER_DWT_WINDOW * (2 ** EMBEDER_DWT_TIMES)
    assert y >= WATER_MARK_Y * EMBEDER_DWT_WINDOW * (2 ** EMBEDER_DWT_TIMES)
    x_multiple = x // (WATER_MARK_X * EMBEDER_DWT_WINDOW)
    y_multiple = y // (WATER_MARK_Y * EMBEDER_DWT_WINDOW)
    min_multiple = min(x_multiple, y_multiple)
    COMPUTE_DWT_TIMES = int(np.log2(min_multiple))
    # 如果计算出来小波变换级数比配置文件中的多
    # 那么最终的小波变换级数就用计算出来的
    # 否则使用配置文件的小波变换级数
    if COMPUTE_DWT_TIMES > EMBEDER_DWT_TIMES:
        DWT_TIMES = COMPUTE_DWT_TIMES
    else:
        DWT_TIMES = EMBEDER_DWT_TIMES

    seed_wm = int(imgNumber[:4])
    seed_dct = int(imgNumber[7:11])
    logger.debug(f'seed_wm:{seed_wm}')
    logger.debug(f'seed_dct:{seed_dct}')

    embedder = DWTWatermarkEmbeder(
        seed_wm,seed_dct,EMBEDER_DWT_MOD_1,
        block_shape=(EMBEDER_DWT_WINDOW,EMBEDER_DWT_WINDOW),
        dwt_deep=DWT_TIMES,
    )
    # 用当前时间生成水印
    now = arrow.now()
    now_text = now.format()
    today = now.format("YYYY-MM-DD")
    wm,md5 = embedder.generate_wm(now_text)

    if not os.path.exists(f'tmp/{today}'):
        os.mkdir(f'tmp/{today}')
    output_file_path = os.path.join('tmp',today,f'{imgNumber}.png')
    output_wm_path = os.path.join('tmp',today,f'wm-{imgNumber}.png')
    # 嵌入水印
    embed_image = embedder.embed(image,wm_img=wm,filename=output_file_path)
    # 生成水印hash
    img_hash = generate_img_hash(embed_image)
    # 水印写盘
    cv.imwrite(output_wm_path,wm)
    # 计算嵌入水印后图片的特征值
    json_str = compartor.extract(embed_image)
    # 更新数据库
    mysql_client.exexcute_update(
       f"""
        UPDATE 
            `{DB_TABLE}` 
        SET 
            random_seed_1 = {seed_wm},
            random_seed_2 = {seed_dct},
            mod_1 = {EMBEDER_DWT_MOD_1},
            dwt_times = {DWT_TIMES},
            features = '{json_str}'
        WHERE `imgNumber`='{imgNumber}'
        """
    )

    return output_file_path,mysql_client,md5,img_hash,imgNumber


def upload_filepath_to_cos(obj):
    output_file_path = obj.result()[0]
    mysql_client = obj.result()[1]
    md5 = obj.result()[2]
    img_hash = obj.result()[3]
    imgNumber = obj.result()[4]

    start = time.time()
    # 上传嵌入水印后的原图
    cos.upload(output_file_path,md5)
    end = time.time()
    logger.info(f"嵌入水印图片{output_file_path}上传成功,上传耗时:{end - start}")

    mysql_client.exexcute_update(
        f"""
         UPDATE 
             `{DB_TABLE}` 
         SET 
             waterMark = 1,
             waterhash = '{img_hash}',
             waterNumber ='{md5}' 
         WHERE `imgNumber`='{imgNumber}'
         """
    )


def generate_img_hash(img):
    start = time.time()
    # 图片转hash
    hash = hashlib.sha256(img.tostring()).hexdigest()
    # np.fromstring(img_str)
    end = time.time()
    logger.info(f"生成图片哈希码成功,耗时:{end - start}")
    return hash


def upload_file_to_cos(img,key,type='.jpg'):
    img_encode = cv.imencode(type, img)
    str_encode = img_encode[1].tostring()
    buf_str = BytesIO(str_encode).getvalue()
    Etag = cos.load_bytes(buf_str,key)


def main():
    if args.mode == 'run':
        try:
            logger.info("当前模式不是debug模式，启动调度器")
            compartor = Compartor()
            mysql_client = MysqlClient()
            pool = ThreadPoolExecutor(max_workers=THREAD_POOL_MAX_WORKER, thread_name_prefix=THREAD_NAME_PREFIX)

            scheduler = BlockingScheduler()
            scheduler.add_job(query_img_to_wm, 'interval', minutes=SD_QUERY_INTERVAL,args=[mysql_client, compartor, pool,])
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            pool.shutdown(wait=True)
            scheduler.shutdown(wait=True)
            logger.debug("程序异常退出")

    if args.mode == 'debug':
        # -------------------------------------------------------------------------------------------------------------- #
        if args.gw:
            # TODO ： 调整水印大小
            if args.name and args.id:
                # python eye.py debug -g  --name 差不多 --id 312391432483842
                logger.info(f"开始测试生成水印,要生成水印的姓名:[{args.name}] ,身份证号:[{args.id}]")
                wm = DWTWatermarkEmbeder.generate_wm(args.name,args.id)
                if args.output_path:
                    if os.path.isdir(args.output_path):
                        output_path = os.path.join(args.output_path,'debug-generate-watermark.png')
                        if os.path.exists(output_path):
                            os.remove(output_path)
                            cv.imwrite(output_path,wm)
                        else:
                            cv.imwrite(output_path, wm)
                    else:
                        raise Exception("参数--output_path必须是一个存在的文件夹!")
                else:
                    output_path = './debug-output'
                    if not os.path.exists(output_path):
                        os.mkdir(output_path)
                    output_path = os.path.join(output_path,'debug-generate-watermark.png')
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    cv.imwrite(output_path, wm)
                    cv.imshow('x',wm)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
            logger.info("水印生成完毕。")
        # -------------------------------------------------------------------------------------------------------------- #
        if args.embed:
            logger.debug("开始测试对图片嵌入水印..")
            if not args.original_file_path:
                raise Exception("请在--original_file_path参数中提供要嵌入水印的原图路径！")
            else:
                if not os.path.isfile(args.original_file_path):
                    raise Exception("--original_file_path参数必须指向某个文件!")
                else:
                    if not args.original_file_path.endswith(".jpg") and not args.original_file_path.endswith(".png"):
                        logger.warn("--original_file_path参数指向的文件不是.jpg或者.png文件！")

            if not args.watermark_file_path:
                raise Exception("请在--watermark_file_path参数中提供要嵌入的水印图片路径！")
            else:
                if not os.path.isfile(args.watermark_file_path):
                    raise Exception("--watermark_file_path参数必须指向某个文件!")
                else:
                    if not args.watermark_file_path.endswith(".jpg") and not args.watermark_file_path.endswith(".png"):
                        logger.warn("--watermark_file_path参数指向的文件不是.jpg或者.png文件！")

            original_image = cv.imread(args.original_file_path)
            water_mark = cv.imread(args.watermark_file_path)

            assert original_image.shape[1] >= water_mark.shape[1] * EMBEDER_DWT_WINDOW * (2 ** EMBEDER_DWT_TIMES)
            assert original_image.shape[0] >= water_mark.shape[0] * EMBEDER_DWT_WINDOW * (2 ** EMBEDER_DWT_TIMES)
            x_multiple = original_image.shape[1] // (water_mark.shape[1] * EMBEDER_DWT_WINDOW)
            y_multiple = original_image.shape[0] // (water_mark.shape[0] * EMBEDER_DWT_WINDOW)
            min_multiple = min(x_multiple,y_multiple)
            COMPUTE_DWT_TIMES = int(np.log2(min_multiple))

            # 如果计算出来小波变换级数比配置文件中的多
            # 那么最终的小波变换级数就用计算出来的
            # 否则使用配置文件的小波变换级数
            if COMPUTE_DWT_TIMES > EMBEDER_DWT_TIMES:
                DWT_TIMES = COMPUTE_DWT_TIMES
            else:
                DWT_TIMES = EMBEDER_DWT_TIMES

            embedder = DWTWatermarkEmbeder(
                4399, 2333, 64,
                dwt_deep=DWT_TIMES
            )
            embedded_image = embedder.embed(ori_img=original_image, wm_img=water_mark)
            if args.output_path:
                if os.path.isdir(args.output_path):
                    output_path = os.path.join(args.output_path, 'debug-embed-watermark.png')
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        cv.imwrite(output_path, embedded_image)
                    else:
                        cv.imwrite(output_path, embedded_image)
                else:
                    raise Exception("参数--output_path必须是一个存在的文件夹!")
            else:
                output_path = './debug-output'
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                output_path = os.path.join(output_path, 'debug-embed-watermark.png')
                if os.path.exists(output_path):
                    os.remove(output_path)
                cv.imwrite(output_path, embedded_image)
                # cv.imshow('x',wm)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
        # -------------------------------------------------------------------------------------------------------------- #
        if args.extract:
            logger.debug("开始测试提取图片中的水印..")
            if not args.extract_file_path:
                raise Exception("请在--extract_file_path参数中提供要嵌入水印的原图路径！")
            else:
                if not os.path.isfile(args.extract_file_path):
                    raise Exception("--extract_file_path参数必须指向某个文件!")
                else:
                    if not args.extract_file_path.endswith(".jpg") and not args.extract_file_path.endswith(".png"):
                        logger.warn("--extract_file_path参数指向的文件不是.jpg或者.png文件！")

            extract_file = cv.imread(args.extract_file_path)
            assert extract_file.shape[1] >= WATER_MARK_X * EMBEDER_DWT_WINDOW * (2 ** EMBEDER_DWT_TIMES)
            assert extract_file.shape[0] >= WATER_MARK_Y * EMBEDER_DWT_WINDOW * (2 ** EMBEDER_DWT_TIMES)
            x_multiple = extract_file.shape[1] // (WATER_MARK_X * EMBEDER_DWT_WINDOW)
            y_multiple = extract_file.shape[0] // (WATER_MARK_Y * EMBEDER_DWT_WINDOW)
            min_multiple = min(x_multiple, y_multiple)
            COMPUTE_DWT_TIMES = int(np.log2(min_multiple))
            # 如果计算出来小波变换级数比配置文件中的多
            # 那么最终的小波变换级数就用计算出来的
            # 否则使用配置文件的小波变换级数
            if COMPUTE_DWT_TIMES > EMBEDER_DWT_TIMES:
                DWT_TIMES = COMPUTE_DWT_TIMES
            else:
                DWT_TIMES = EMBEDER_DWT_TIMES
            extractor = DwtWatermarkExractor(
                4399,2333,64,
                wm_shape=(WATER_MARK_Y,WATER_MARK_X),
                dwt_deep=DWT_TIMES
            )
            wm = extractor.extract(extract_file)
            if args.output_path:
                if os.path.isdir(args.output_path):
                    output_path = os.path.join(args.output_path, 'debug-extract-watermark.png')
                    if os.path.exists(output_path):
                        os.remove(output_path)
                        cv.imwrite(output_path, wm)
                    else:
                        cv.imwrite(output_path, wm)
                else:
                    raise Exception("参数--output_path必须是一个存在的文件夹!")
            else:
                output_path = './debug-output'
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                output_path = os.path.join(output_path, 'debug-extract-watermark.png')
                if os.path.exists(output_path):
                    os.remove(output_path)
                cv.imwrite(output_path, wm)



if __name__ == '__main__':
    main()

