# -*- coding: utf-8 -*-

import base64
import numpy as np
from PIL import Image
import cv2
# import matplotlib.pyplot as plt


class ImageUtils:
    # 读取磁盘图片文件, 并将其转换为NumPy的ndarray类型
    @staticmethod
    def load_image_as_ndarray(file_path):
        img = Image.open(file_path)
        img_array = np.array(img)
        return img_array

    # 将图片base64字符串转成ndarray矩阵类型
    @staticmethod
    def image_base64_to_ndarray(image_base64_encode:str):
        img_base64_decode = base64.b64decode(image_base64_encode)
        img_array = np.fromstring(img_base64_decode, np.uint8)
        return cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)

    @staticmethod
    def img_ndarray_to_base64(img_array):
        #img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        encode_image = cv2.imencode(".jpg", img_array)[1]
        byte_data = encode_image.tobytes()
        base64_str = base64.b64encode(byte_data).decode("ascii")
        return base64_str

    # 图片文件content转cv2，并缩放到指定尺寸
    @staticmethod
    def content_to_cv2(contents: list, size: tuple):
        '''
        content -> np -> cv2 -> cv2<target_size>'''
        imgs_np = [np.asarray(bytearray(content), dtype=np.uint8) for content in contents]
        imgs_cv2 = [cv2.imdecode(img_np, cv2.IMREAD_COLOR) for img_np in imgs_np]
        imgs_cv2 = [cv2.resize(img_cv2, size, interpolation=cv2.INTER_LINEAR) for img_cv2 in imgs_cv2]
        return imgs_cv2

    @staticmethod
    def base64_to_cv2(img: str):
        # 注：仅适合图像，不适合其它numpy数组，例如bboxs(人脸标注框)的数据
        # base64 -> 二进制 -> ndarray -> cv2
        # 解码为二进制数据
        img_codes = base64.b64decode(img)
        img_np = np.frombuffer(img_codes, np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        return img_cv2

    @staticmethod
    def cv2_to_base64(image):
        data = cv2.imencode('.jpg', image)[1]
        return base64.b64encode(data.tostring()).decode('utf8')

    @staticmethod
    def np_to_base64(array):
        return base64.b64encode(array.tostring()).decode('utf8')

    @staticmethod
    def base64_to_np(arr_b64):
        return np.frombuffer(base64.b64decode(arr_b64), np.float32)

    # 显示cv2格式的图像
    @staticmethod
    def cv2_show(img_cv2):
        cv2.imshow('img', img_cv2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 画人脸标注框
    @staticmethod
    def cv2_with_rectangle(img_cv2, bboxs: list):
        '''return --> 画好矩形标注框的图像'''
        bboxs = [bbox.astype('int32') for bbox in bboxs]
        for bbox in bboxs:
            cv2.rectangle(
                img_cv2,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (255, 0, 0),  # 蓝色
                thickness=2)
        return img_cv2

    # 计算特征向量的余弦相似度
    @staticmethod
    def compare_face(emb1: np.ndarray, emb2: np.ndarray, threshold=0.6):
        '''
        @return -> (<numpy.bool>, <numpy.float32>)
        - bool: 是否为同一张人脸
        - float: 余弦相似度[-1, 1]，值越大越相似 \n
        @params
        - threshold: 判断两张人脸为同一张的余弦相似度阈值
        '''
        # return --> 余弦相似度[-1, 1]，值越大，越相似
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return sim > threshold, sim