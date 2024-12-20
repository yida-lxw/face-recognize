import cv2

from utils.FaceRecognitionUtil import FaceRecognition


def main():
    face_recognition = FaceRecognition()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception('摄像头打开失败')
    while True:
        ret, frame = cap.read()
        if ret:
            # 人脸识别
            # results = face_recognition.recognition(image=frame)
            # img = face_recognition.draw_recognition(image=frame, results=results)

            # 人脸属性识别
            results = face_recognition.face_attribute(image=frame)
            img = face_recognition.draw_attribute(image=frame, results=results, draw_2d_landmark=True)

            cv2.imshow("result", img)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
