import os

import cv2
import numpy as np

from utils.FaceRecognitionUtil import FaceRecognition

if __name__ == '__main__':
    face_recognition = FaceRecognition()
    # 人脸注册
    img = cv2.imdecode(np.fromfile('face_db/longgang/longgang.jpg', dtype=np.uint8), -1)
    result, state = face_recognition.register(image=img, user_name='longgang')
    if not result:
        print(state)
    # img = cv2.imdecode(np.fromfile('images/dilireba_08.jpg', dtype=np.uint8), -1)
    # result, state = face_recognition.register(image=img, user_name='dilireba')
    # if not result:
    #     print(state)

    # 人脸识别
    img = cv2.imdecode(np.fromfile('images/longgang_01.jpg', dtype=np.uint8), -1)
    results = face_recognition.recognition(image=img)
    img = face_recognition.draw_recognition(image=img, results=results)
    cv2.imshow("result", img)
    cv2.waitKey(0)

    # 人脸属性识别
    # img = cv2.imdecode(np.fromfile('test.jpg', dtype=np.uint8), -1)
    # results = face_recognition.face_attribute(image=img)
    # img = face_recognition.draw_attribute(image=img, results=results, draw_2d_landmark=True)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
