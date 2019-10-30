# -*- coding: utf-8 -*-
# Bucket由bucketname-appid组成
# ETag HTTP响应头是资源的特定版本的标识符。这可以让缓存更高效，并节省带宽。
# 因为如果内容没有改变，Web服务器不需要发送完整的响应。
# 而如果内容发生了变化，使用ETag有助于防止资源的同时更新相互覆盖
import os
import sys
import cv2 as cv
import numpy as np
from qcloud_cos import CosConfig
from qcloud_cos import CosS3Client
from qcloud_cos import CosServiceError
from qcloud_cos import CosClientError
from fddeye.tools import logger
from io import BytesIO
from fddeye.config import *


try:
    config = CosConfig(Region=COS_REGION, SecretId=COS_SECRET_ID,
       SecretKey=COS_SECRET_KEY, Token=COS_TOKEN)
    cos_client = CosS3Client(config)
except CosClientError as e:
    logger.error(e)



def load_file(file_name,local=False):
    if local:
        load_local_file(file_name)
    else:
        with open(file_name, 'rb') as fp:
            response = cos_client.put_object(
                Bucket=COS_BUCKET,
                Body=fp,
                Key=file_name,
                StorageClass='STANDARD',
                ContentType='text/html; charset=utf-8',
                Metadata={
                    'x-cos-meta-key1': 'value1',
                    'x-cos-meta-key2': 'value2'
                }
            )
            # print(response['ETag'])


def load_local_file(file_name):
    response = cos_client.put_object_from_local_file(
        Bucket=COS_BUCKET,
        LocalFilePath=file_name,
        Key=file_name,
    )
    # print(response['ETag'])


def load_bytes(string,key):
    # 字节流 简单上传
    if not isinstance(string,bytes) and not isinstance(string,BytesIO):
        string = string.encode('utf-8')
    response = cos_client.put_object(
        Bucket=COS_BUCKET,
        Body=string,
        Key=key,
        ContentType='image/jpg',
        # Metadata={
        #     'x-cos-meta-key1': 'value1',
        #     'x-cos-meta-key2': 'value2'
        # }
    )
    return response['ETag']



def upload(file_path,key):
    response = cos_client.upload_file(
        Bucket=COS_BUCKET,
        LocalFilePath=file_path,
        Key=key,
        PartSize=10000,
        MAXThread=10
    )
    return response['ETag']


def get_img_file(key,chunk=COS_DOWNLOAD_IMG_CHUNK,save=False,save_path=None,ETag='',rang=''):
    try:
        total_size = get_object_meta(key)['Content-Length']
        response = cos_client.get_object(
            Bucket=COS_BUCKET,
            Key=key,
            Range=rang,  # 指定下载范围
            IfMatch=ETag, # ETag 与指定的内容一致时才返回
            ResponseContentEncoding='utf-8',
            ResponseContentLanguage='zh-cn',
            ResponseContentType='image/png',  # 设置Response HTTP 头部
            # ResponseContentType='text/html; charset=utf-8',  # 设置Response HTTP 头部
        )
    except CosServiceError as e:
        logger.error(e.get_origin_msg())
        logger.error(e.get_digest_msg())
        logger.error(e.get_status_code())
        logger.error(e.get_error_code())
        logger.error(e.get_error_msg())
        logger.error(e.get_resource_location())
        logger.error(e.get_trace_id())
        logger.error(e.get_request_id())

    if save and save_path is None:
        logger.warn("存储路径未填写，系统默认存储到当前目录/download")
        if not os.path.exists("./download"):
            os.mkdir("./download")
        response['Body'].get_stream_to_file("./download")
    elif save and save_path:
        logger.info(f"正在将文件[{key}]写入路径[{save_path}]")
        response['Body'].get_stream_to_file(save_path)

    fp = response['Body'].get_raw_stream()
    image_bytes = b""
    current_size = 0
    block = fp.read(chunk)
    while block:
        current_size += len(block)
        image_bytes += block
        rate = int(100 * (float(current_size) / float(total_size)))
        print(f'\r当前文件{key}的下载进度:{rate}%',end=' ') # \r:主要功能是将光标回到一行的开始位置；
        if len(block) < chunk : print('')
        sys.stdout.flush()
        block = fp.read(chunk)

    # 对图片解码
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv.imdecode(image_array,cv.COLOR_RGB2BGR)
    return response,image


def delete_single_obj(key):
    try:
        result_dict = cos_client.delete_object(
            Bucket=COS_BUCKET,
            Key=key
        )
    except CosServiceError as e:
        logger.error(e.get_origin_msg())
        logger.error(e.get_digest_msg())
        logger.error(e.get_status_code())
        logger.error(e.get_error_code())
        logger.error(e.get_error_msg())
        logger.error(e.get_resource_location())
        logger.error(e.get_trace_id())
        logger.error(e.get_request_id())
    return result_dict


def get_object_meta(key):
    try:
        result_dict = cos_client.head_object(
            Bucket=COS_BUCKET,
            Key=key
        )
    except CosServiceError as e:
        logger.error(e.get_origin_msg())
        logger.error(e.get_digest_msg())
        logger.error(e.get_status_code())
        logger.error(e.get_error_code())
        logger.error(e.get_error_msg())
        logger.error(e.get_resource_location())
        logger.error(e.get_trace_id())
        logger.error(e.get_request_id())
    return result_dict


# 慎用此方法
def list_objs():
    while True:
        try:
            response = cos_client.list_objects(
                Bucket=COS_BUCKET,
            )
            if 'Contents' not in response.keys():
                logger.error("目前存储桶可能是空的")
                break
            for dict_ in response['Contents']: # [dict,dict,dict ...]
                print(dict_)
            if response['IsTruncated'] == 'false':
                break
        except CosServiceError as e:
            logger.error(e.get_origin_msg())
            logger.error(e.get_digest_msg())
            logger.error(e.get_status_code())
            logger.error(e.get_error_code())
            logger.error(e.get_error_msg())
            logger.error(e.get_resource_location())
            logger.error(e.get_trace_id())
            logger.error(e.get_request_id())
            break

# upload('mark/images/lena.png',key='lena.png')
# get_img_file('xiaozhi')
