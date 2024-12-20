
import json
import cv2
import numpy as np
from flasgger import Swagger
from flask import Flask, request, jsonify
from flask_cors import CORS

from ComplexEncoder import ComplexEncoder
from ImageUtils import ImageUtils
from utils.FaceRecognitionUtil import FaceRecognition
from utils.logger import setup_logger

logger = setup_logger('server')

# 读取服务配置文件
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 允许跨越访问
CORS(app)
# 文档配置
app.config['SWAGGER'] = {
    'title': '人脸识别服务',
    'uiversion': 3
}
Swagger(app)

# 加载人脸识别工具
face_recognition = FaceRecognition()


@app.route('/detect', methods=['POST'])
def detect_faces():
    img_base64 = request.form.get('image_base64')
    return face_recognition.detect(img_base64)


@app.route('/match_two_faces_image_base64', methods=['POST'])
def match_two_faces_image_base64():
    image_live_face_base_64 = request.form.get('live_face')
    image_registe_face_base_64 = request.form.get('registe_face')
    if ((image_live_face_base_64 is None or len(image_live_face_base_64) <= 0) or
            (image_registe_face_base_64 is None or len(image_registe_face_base_64) <= 0)):
        return_data_dict = dict()
        return_data_dict['code'] = 500
        return_data_dict['msg'] = "failed"
        return_data_dict['is_like'] = False
        return_data_dict['score'] = 0
        return jsonify(return_data_dict)

    image_live_face_base_64 = fix_base64(image_live_face_base_64)
    image_registe_face_base_64 = fix_base64(image_registe_face_base_64)
    imgs_base64 = [image_live_face_base_64, image_registe_face_base_64]

    # 4. 载入模型 --> 获得特征向量 + 人脸标注框
    rs = [face_recognition.detect(img_base64=img_base64) for img_base64 in imgs_base64]
    embeddings = [r_json['embeddings'] for r_json in rs]
    embeddings = [[ImageUtils.base64_to_np(emb) for emb in embs] for embs in embeddings]
    #bboxs = [r_json['bboxs'] for r_json in rs_json]
    #bboxs = [[ImageUtils.base64_to_np(bbox) for bbox in bs] for bs in bboxs]

    # 5. 比较两张图片中，各自第一张人脸的特征向量
    embs = [embeddings[i][0] for i in range(len(embeddings))]
    is_like, how_like = ImageUtils.compare_face(embs[0], embs[1], threshold=0.5)

    # 6. 框出检测到的人脸(第一张)
    # imgs_cv2 = [ImageUtils.cv2_with_rectangle(imgs_cv2[i], bboxs[i]) for i in range(len(imgs_cv2))]
    # imgs_base64 = [ImageUtils.cv2_to_base64(img_cv2) for img_cv2 in imgs_cv2]

    # 7. 返回比较结果
    return_data_dict = dict()
    return_data_dict['code'] = 200
    return_data_dict['msg'] = "success"
    return_data_dict['is_like'] = is_like
    return_data_dict['score'] = how_like
    return json.dumps(return_data_dict, cls=ComplexEncoder)

@app.route('/match_two_faces_image_filepaths', methods=['POST'])
def match_two_faces_image_filepaths():
    image_live_face = request.form.get('live_face')
    image_registe_face = request.form.get('registe_face')
    if (image_live_face is None or len(image_live_face) <= 0) or (image_registe_face is None or len(image_registe_face) <= 0):
        return_data_dict = dict()
        return_data_dict['code'] = 500
        return_data_dict['msg'] = "failed"
        return_data_dict['is_like'] = False
        return_data_dict['score'] = 0
        return jsonify(return_data_dict)
    image_faces = [image_live_face, image_registe_face]
    # 2. 图片文件转cv2，并缩放到指定尺寸
    imgs_cv2 = [ImageUtils.load_image_as_ndarray(image_face_path) for image_face_path in image_faces]

    # 3. cv2转base64编码的字符串 --> 传给模型
    imgs_base64 = [ImageUtils.cv2_to_base64(img_cv2) for img_cv2 in imgs_cv2]

    # 4. 载入模型 --> 获得特征向量 + 人脸标注框
    rs = [face_recognition.detect(img_base64) for img_base64 in imgs_base64]
    embeddings = [r_json['embeddings'] for r_json in rs]
    embeddings = [[ImageUtils.base64_to_np(emb) for emb in embs] for embs in embeddings]
    #bboxs = [r_json['bboxs'] for r_json in rs_json]
    #bboxs = [[ImageUtils.base64_to_np(bbox) for bbox in bs] for bs in bboxs]

    # 5. 比较两张图片中，各自第一张人脸的特征向量
    embs = [embeddings[i][0] for i in range(len(embeddings))]
    is_like, how_like = ImageUtils.compare_face(embs[0], embs[1], threshold=0.5)

    # 6. 框出检测到的人脸(第一张)
    # imgs_cv2 = [ImageUtils.cv2_with_rectangle(imgs_cv2[i], bboxs[i]) for i in range(len(imgs_cv2))]
    # imgs_base64 = [ImageUtils.cv2_to_base64(img_cv2) for img_cv2 in imgs_cv2]

    # 7. 返回比较结果
    return_data_dict = dict()
    return_data_dict['code'] = 200
    return_data_dict['msg'] = "success"
    return_data_dict['is_like'] = is_like
    return_data_dict['score'] = how_like
    return json.dumps(return_data_dict, cls=ComplexEncoder)

@app.route('/match_two_faces_image_files', methods=['POST'])
def match_two_faces_image_files():
    files = request.files.getlist("image")
    if files is None or len(files) != 2:
        return_data_dict = dict()
        return_data_dict['code'] = 500
        return_data_dict['msg'] = "failed"
        return_data_dict['is_like'] = False
        return_data_dict['score'] = 0
        return jsonify(return_data_dict)
    file1 = files[0]
    file2 = files[1]
    files = [file1, file2]
    contents = [file.read() for file in files]
    # 2. 图片文件转cv2，并缩放到指定尺寸
    imgs_cv2 = ImageUtils.content_to_cv2(contents, face_recognition.get_default_Scale_size())

    # 3. cv2转base64编码的字符串 --> 传给模型
    imgs_base64 = [ImageUtils.cv2_to_base64(img_cv2) for img_cv2 in imgs_cv2]

    # 4. 载入模型 --> 获得特征向量 + 人脸标注框
    rs = [face_recognition.detect(img_base64) for img_base64 in imgs_base64]
    embeddings = [r_json['embeddings'] for r_json in rs]
    embeddings = [[ImageUtils.base64_to_np(emb) for emb in embs] for embs in embeddings]
    #bboxs = [r_json['bboxs'] for r_json in rs_json]
    #bboxs = [[ImageUtils.base64_to_np(bbox) for bbox in bs] for bs in bboxs]

    # 5. 比较两张图片中，各自第一张人脸的特征向量
    embs = [embeddings[i][0] for i in range(len(embeddings))]
    is_like, how_like = ImageUtils.compare_face(embs[0], embs[1], threshold=0.5)

    # 6. 框出检测到的人脸(第一张)
    # imgs_cv2 = [ImageUtils.cv2_with_rectangle(imgs_cv2[i], bboxs[i]) for i in range(len(imgs_cv2))]
    # imgs_base64 = [ImageUtils.cv2_to_base64(img_cv2) for img_cv2 in imgs_cv2]

    # 7. 返回比较结果
    return_data_dict = dict()
    return_data_dict['code'] = 200
    return_data_dict['msg'] = "success"
    return_data_dict['is_like'] = is_like
    return_data_dict['score'] = how_like
    return json.dumps(return_data_dict, cls=ComplexEncoder)


# 人脸注册或者添加人脸
@app.route('/register_image', methods=['POST'])
def register_image():
    user_name = request.form.get('user_name')
    if not user_name or len(user_name) <= 0:
        logger.info('注册的人脸所属用户名称不能为空')
        return_data_dict = dict()
        return_data_dict['code'] = 500
        return_data_dict['msg'] = "用户名称不能为空"
        return jsonify(return_data_dict)
    # 获取上传图片
    try:
        upload_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f'上传文件不存在，或是图片错误！错误信息e：{e}')
        return jsonify({"code": 500, "msg": "上传文件不存在，或是图片错误"})

    # 执行人脸注册或者添加人脸
    result, state = face_recognition.register_image(image=image, user_name=user_name)
    if not result:
        logger.warning(f'执行人脸注册或者添加人脸失败！错误信息e：{state}')
        return jsonify({"msg": "执行人脸注册或者添加人脸失败"})

    # 添加返回信息
    return_data_dict = dict()
    return_data_dict['code'] = 200
    return_data_dict['msg'] = "success"
    return jsonify(return_data_dict)

# 人脸注册或者添加人脸
@app.route('/register_image_base64', methods=['POST'])
def register_image_base64():
    user_name = request.form.get('user_name')
    if not user_name or len(user_name) <= 0:
        logger.info('注册的人脸所属用户名称不能为空')
        return_data_dict = dict()
        return_data_dict['code'] = 500
        return_data_dict['msg'] = "用户名称不能为空"
        return jsonify(return_data_dict)
    image_base64 = request.form.get('image_base64')
    if not image_base64 or len(image_base64) <= 0:
        logger.info('人脸图片base64字符串不能为空')
        return_data_dict = dict()
        return_data_dict['code'] = 500
        return_data_dict['msg'] = "人脸图片base64字符串不能为空"
        return jsonify(return_data_dict)

    # 执行人脸注册或者添加人脸
    result, state = face_recognition.register_image_base64(image_base64=image_base64, user_name=user_name)
    if not result:
        logger.warning(f'执行人脸注册或者添加人脸失败！错误信息e：{state}')
        return jsonify({"code": 500, "msg": "执行人脸注册或者添加人脸失败"})

    # 添加返回信息
    return_data_dict = dict()
    return_data_dict['code'] = 200
    return_data_dict['msg'] = "success"
    return jsonify(return_data_dict)


# 人脸识别
@app.route('/recognition', methods=['POST'])
def recognition():
    # 获取上传图片
    try:
        # multipart/form-data
        upload_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f'上传文件不存在，或是图片错误！错误信息e：{e}')
        return jsonify({"msg": "上传文件不存在，或是图片错误"})

    # 执行人脸识别
    results = face_recognition.recognition(image=image)
    bbox = results[0]["bbox"]
    user_name = results[0]["user_name"]
    unmatched = (bbox is None or len(bbox) <= 0) or (user_name is None or len(user_name) <= 0)
    # 添加返回信息
    return_data_dict = dict()
    return_data_dict['code'] = 500 if unmatched else 200
    return_data_dict['msg'] = "failed" if unmatched else "success"
    return_data_dict['results'] = results
    return jsonify(return_data_dict)


# 人脸属性识别
@app.route('/face_attribute', methods=['POST'])
def face_attribute():
    # 获取上传图片
    try:
        upload_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(upload_file.read(), np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.warning(f'上传文件不存在，或是图片错误！错误信息e：{e}')
        return jsonify({"msg": "上传文件不存在，或是图片错误"})

    # 执行人脸属性识别
    results = face_recognition.face_attribute(image=image)

    # 添加返回信息
    return_data_dict = dict()
    return_data_dict['msg'] = "success"
    return_data_dict['results'] = results
    return jsonify(return_data_dict)

def fix_base64(image_base64:str):
    image_base64 = image_base64.replace("%2F", "/")
    image_base64 = image_base64.replace("%2B", "+")
    image_base64 = image_base64.replace("%20", " ")
    image_base64 = image_base64.replace("%3F", "?")
    image_base64 = image_base64.replace("%25", "%")
    image_base64 = image_base64.replace("%26", "&")
    image_base64 = image_base64.replace("%3D", "=")
    return image_base64


if __name__ == '__main__':
    print("server start...")
    app.run(host='0.0.0.0', port=9099, debug=False, threaded=True)
