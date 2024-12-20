import os
import pickle

import cv2
import insightface
import numpy as np
from insightface.app.common import Face
from insightface.utils import face_align
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

from ImageUtils import ImageUtils
from file_utils import FileUtils
from sql_helper import find_by_id, find_one, find_many, insert, update
from utils.logger import setup_logger

logger = setup_logger('FaceRecognition')


class FaceRecognition:
    def __init__(self, gpu_id=0, model_type='l', face_db='face_db', threshold=0.40, det_thresh=0.5,
                 det_size=(640, 640), face_indexes_path='resource/face_indexes.bin'):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param model_type: 模型类型，包含l、m、s
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        # 索引候选数量
        self.cdd_num = 5
        self.batch_size = 32
        self.face_db = face_db
        self.model_type = model_type
        self.threshold = threshold
        self.face_indexes_path = face_indexes_path
        self.font_style = None
        # 人脸库的人脸特征
        self.faces_feature = None
        # 人脸特征对应的用户名
        self.users_name = []
        # 人脸特征对应的人脸文件路径
        self.users_image_path = []

        assert model_type in ['l', 'm', 's'], "模型类型错误，模型类型应该输入['l', 'm', 's']"
        # 加载人脸识别模型
        self.model = insightface.app.FaceAnalysis(name=f'buffalo_{model_type}',
                                                  root='./',
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=gpu_id, det_thresh=det_thresh, det_size=det_size)
        # 加载人脸库中的人脸
        self.__load_faces(self.face_db)

    # 加载人脸特征索引
    def __load_face_indexes(self):
        # 如果存在人脸特征索引文件就加载
        if not os.path.exists(self.face_indexes_path): return
        with open(self.face_indexes_path, "rb") as f:
            indexes = pickle.load(f)
        model_type = indexes["model_type"]
        # 必须要保证是同一个模型输出的人脸特征
        if model_type == self.model_type:
            self.users_name = indexes["users_name"]
            self.faces_feature = indexes["faces_feature"]
            self.users_image_path = indexes["users_image_path"]
        else:
            logger.warning("使用了不同模型，将重新生成人脸索引库")

    # 保存人脸特征索引
    def __write_index(self):
        with open(self.face_indexes_path, "wb") as f:
            pickle.dump({"users_name": self.users_name,
                         "faces_feature": self.faces_feature,
                         "users_image_path": self.users_image_path,
                         "model_type": self.model_type}, f)

    # 加载人脸库中的人脸
    def __load_faces(self, face_db_path):
        # 先加载人脸特征索引
        self.__load_face_indexes()
        os.makedirs(face_db_path, exist_ok=True)
        images_path = []
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                images_path.append(os.path.join(root, file).replace('\\', '/'))
        # 人脸库没数据就跳过
        if len(images_path) == 0: return
        logger.info('正在加载人脸库数据...')
        input_images = []
        for image_path in tqdm(images_path):
            # 如果人脸特征已经在索引就跳过
            if image_path in self.users_image_path: continue
            # 读取人脸图片
            input_image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), 1)
            # 获取用户名
            user_name = os.path.basename(os.path.dirname(image_path))
            self.users_name.append(user_name)
            self.users_image_path.append(image_path)
            input_images.append(input_image)
            # 处理一批数据
            if len(input_images) == self.batch_size:
                embeddings = self.model.models['recognition'].get_feat(input_images)
                if self.faces_feature is None:
                    self.faces_feature = embeddings
                else:
                    self.faces_feature = np.vstack((self.faces_feature, embeddings))
                input_images = []
        # 处理不满一批的数据
        if len(input_images) != 0:
            embeddings = self.model.models['recognition'].get_feat(input_images)
            if self.faces_feature is None:
                self.faces_feature = embeddings
            else:
                self.faces_feature = np.vstack((self.faces_feature, embeddings))
        assert len(self.faces_feature) == len(self.users_name) == len(self.users_image_path), '加载的数量对不上！'
        # 将人脸特征保存到索引文件中
        self.__write_index()
        logger.info('人脸库数据加载完成！')

    # 注册或者添加人脸
    def register_image(self, image, user_name:str, num:int=1):
        if not user_name or len(user_name) <= 0:
            logger.info('注册的人脸所属用户名称不能为空')
            return False, "参数user_name不能为空"
        faces_bbox, faces = self.detection(image=image)
        # 保证注册的图片中只有一个人脸
        if len(faces) != 1:
            logger.info(f'图片中人脸不唯一，注册或添加人脸失败，人脸数量：{len(faces)}')
            return False, "图片中人脸不唯一"
        aimg = face_align.norm_crop(image, landmark=faces[0].kps)
        embedding = self.model.models['recognition'].get_feat(aimg)
        for i, e in enumerate(self.faces_feature):
            if (e.reshape(1, -1) == embedding).all():
                logger.info(f"人脸重复添加，人脸与[{self.users_image_path[i]}]完全相同")
                return False, f"人脸重复添加，人脸与[{self.users_image_path[i]}]完全相同"
        os.makedirs(os.path.join(self.face_db, user_name), exist_ok=True)
        # 保存人脸图片
        image_path = os.path.join(self.face_db, user_name,
                                  f'{len(os.listdir(os.path.join(self.face_db, user_name)))}.png')
        cv2.imencode('.jpg', aimg)[1].tofile(image_path)

        select_sql = "select * from sys_user where username = %s"
        current_sys_user = find_one(sql=select_sql, args=[user_name])
        current_userid = current_sys_user["user_id"]
        update_field = "face"+ str(num)
        update_sql = "update sys_user_verify set {face_field} = ? where user_id = ?".format(face_field=update_field)
        update_sql = (update_sql.replace(update_field + " = ?", update_field + " = %s")
                      .replace("user_id = ?", "user_id = %s"))
        image_base64 = ImageUtils.img_ndarray_to_base64(image)
        effect_row = update(sql=update_sql, args=(image_base64, current_userid))
        if effect_row > 0:
            logger.info(f"update face image for user:{current_userid} successfully.")
            if self.faces_feature is None:
                self.faces_feature = embedding
            else:
                self.faces_feature = np.vstack((self.faces_feature, embedding))
            self.users_name.append(user_name)
            self.users_image_path.append(image_path.replace('\\', '/'))
            # 将人脸特征保存到索引文件中
            self.__write_index()
            return True, "注册成功"
        else:
            return False, "注册失败"

    def register_image_base64(self, image_base64:str, user_name:str, num:int=1):
        if not user_name or len(user_name) <= 0:
            logger.info('注册的人脸所属用户名称不能为空')
            return False, "参数user_name不能为空"
        image_ndarray = ImageUtils.image_base64_to_ndarray(image_base64)
        faces_bbox, faces = self.detection(image=image_ndarray)
        # 保证注册的图片中只有一个人脸
        if len(faces) != 1:
            logger.info(f'图片中人脸不唯一，注册或添加人脸失败，人脸数量：{len(faces)}')
            return False, "图片中人脸不唯一"
        aimg = face_align.norm_crop(image_ndarray, landmark=faces[0].kps)
        embedding = self.model.models['recognition'].get_feat(aimg)
        for i, e in enumerate(self.faces_feature):
            if (e.reshape(1, -1) == embedding).all():
                logger.info(f"人脸重复添加，人脸与[{self.users_image_path[i]}]完全相同")
                return False, f"人脸重复添加，人脸与[{self.users_image_path[i]}]完全相同"
        os.makedirs(os.path.join(self.face_db, user_name), exist_ok=True)
        # 保存人脸图片
        image_path = os.path.join(self.face_db, user_name,
                                  f'{len(os.listdir(os.path.join(self.face_db, user_name)))}.png')
        cv2.imencode('.jpg', aimg)[1].tofile(image_path)

        select_sql = "select * from sys_user where username = %s"
        current_sys_user = find_one(sql=select_sql, args=[user_name])
        current_userid = current_sys_user["user_id"]
        update_field = "face"+ str(num)
        update_sql = "update sys_user_verify set {face_field} = ? where user_id = ?".format(face_field=update_field)
        update_sql = (update_sql.replace(update_field + " = ?", update_field + " = %s")
                      .replace("user_id = ?", "user_id = %s"))
        effect_row = update(sql=update_sql, args=(image_base64, current_userid))
        if effect_row > 0:
            logger.info(f"update face image for user:{current_userid} successfully.")
            if self.faces_feature is None:
                self.faces_feature = embedding
            else:
                self.faces_feature = np.vstack((self.faces_feature, embedding))
            self.users_name.append(user_name)
            self.users_image_path.append(image_path.replace('\\', '/'))
            # 将人脸特征保存到索引文件中
            self.__write_index()
            return True, "注册成功"
        else:
            return False, "注册失败"


    # 清除用户人脸注册信息
    def unRegister(self, image, user_name):
        if not user_name or len(user_name) <= 0:
            logger.info('用户名称不能为空')
            return False, "参数user_name不能为空"
        delete_folder_path = os.path.join(self.face_db, user_name)

        try:
            FileUtils.deleteFolderIfExists(delete_folder_path)
        except Exception as e:
            logger.warning(f'删除用户人脸照片库失败！错误信息e：{e}')
            return False, "取消人脸注册失败"

    # 人脸检索
    def __retrieval(self, np_feature):
        labels = []
        for feature in np_feature:
            similarity = cosine_similarity(self.faces_feature, feature.reshape(1, -1)).squeeze()
            abs_similarity = np.abs(similarity)
            # 获取候选索引
            candidate_idx = np.argpartition(abs_similarity, -self.cdd_num)[-self.cdd_num:]
            # 过滤低于阈值的索引
            remove_idx = np.where(abs_similarity[candidate_idx] < self.threshold)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            candidate_idx_size = candidate_idx.size
            logger.info("candidate_idx_size:" + str(candidate_idx_size))
            if candidate_idx_size <= 0:
                continue
            # 获取标签最多的值
            candidate_label_list = list(np.array(self.users_name)[candidate_idx])
            if len(candidate_label_list) == 0:
                max_label = "unknown"
            else:
                max_label = max(candidate_label_list, key=candidate_label_list.count)
            labels.append(max_label)
        return labels

    # 人脸检测
    def detection(self, image):
        bboxes, kpss = self.model.det_model.detect(image)
        faces = []
        faces_bbox = []
        if bboxes.shape[0] > 0:
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, 0:4]
                det_score = bboxes[i, 4]
                kps = kpss[i]
                # 人脸框、关键点以及检测得分
                face = Face(bbox=bbox, kps=kps, det_score=det_score)
                faces.append(face)
                # 人脸框
                faces_bbox.append(bbox.astype(np.int32).tolist())
        return faces_bbox, faces

    # 人脸识别
    def recognition(self, image):
        # 先检测人脸
        _, faces = self.detection(image=image)
        results, aimgs, np_feature = [], [], []
        bboxes = []
        scores = []
        index = 0;
        for face in faces:
            scores.insert(index, float(face["det_score"]))
            index = index + 1
            bboxes.append(face.bbox.astype(np.int32).tolist())
            # 开始人脸识别
            aimg = face_align.norm_crop(image, landmark=face.kps)
            aimgs.append(aimg)
            # 处理一批数据
            if len(aimgs) == self.batch_size:
                embeddings = self.model.models['recognition'].get_feat(aimgs)
                for embedding in embeddings:
                    np_feature.append(embedding)
                aimgs = []
        # 处理不满一批的数据
        if len(aimgs) != 0:
            embeddings = self.model.models['recognition'].get_feat(aimgs)
            for embedding in embeddings:
                np_feature.append(embedding)
        users = self.__retrieval(np_feature=np_feature)
        if users is None or len(users) <= 0:
            results = [{'bbox': [], 'user_name': "", 'score': 0}]
        else:
            results = [{'bbox': b, 'user_name': u, 'score': s} for b, u, s in zip(bboxes, users, scores)]
        return results

    # 识别人脸属性
    def face_attribute(self, image):
        # 先检测人脸
        _, faces = self.detection(image=image)
        # 获取人脸属性
        results = list()
        for face in faces:
            # 人脸框
            bbox = face.bbox.astype(np.int32).tolist()
            # 年龄性别
            gender, age = self.model.models['genderage'].get(img=image, face=face)
            gender = 'f' if gender == 0 else 'm'
            # 五个关键点
            kps = face.kps.astype(np.int32).tolist()
            # 人脸2D关键点
            landmark_2d_106 = self.model.models['landmark_2d_106'].get(img=image, face=face).astype(np.int32).tolist()
            # 人脸3D关键点
            landmark_3d_68 = self.model.models['landmark_3d_68'].get(img=image, face=face).astype(np.int32).tolist()
            result = {'bbox': bbox, 'gender': gender, 'age': age, 'kps': kps, 'landmark_2d_106': landmark_2d_106,
                      'landmark_3d_68': landmark_3d_68}
            results.append(result)
        return results

    # 画人脸识别结果
    def draw_recognition(self, image, results):
        if self.font_style is None:
            self.font_style = ImageFont.truetype("simsun.ttc", 18, encoding="utf-8")
        image = image.copy()
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for result in results:
            bbox = result["bbox"]
            user = result["user_name"]
            xmin, ymin, xmax, ymax = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            # 画人脸框
            draw.rectangle([xmin, ymin, xmax, ymax], outline=(0, 0, 255), width=2)
            # 画人脸名称
            draw.text((xmin, ymin), user, (0, 255, 0), font=self.font_style)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def detect(self, img_base64:str):
        #img_cv2 = ImageUtils.base64_to_cv2(img_base64)
        img_cv2 = ImageUtils.image_base64_to_ndarray(img_base64)
        faces = self.get_face(img_cv2)
        embeddings = [ImageUtils.np_to_base64(face['embedding']) for face in faces]
        # [x1, y1, x2, y2]左上角和右下角的坐标
        bboxs = [ImageUtils.np_to_base64(face['bbox']) for face in faces]
        return {"embeddings": embeddings, "bboxs": bboxs}

    def get_face(self, img_cv2):
        return self.model.get(img_cv2)

    def get_default_Scale_size(self):
        return (640, 640, );


    # 画人脸属性
    @staticmethod
    def draw_attribute(image, results, draw_2d_landmark=True):
        image = image.copy()
        for result in results:
            if draw_2d_landmark:
                # 画2D关键点
                for l in result["landmark_2d_106"]:
                    cv2.circle(image, (l[0], l[1]), 1, (0, 0, 255), 1)
            else:
                # 画3D关键点
                for l in result["landmark_3d_68"]:
                    cv2.circle(image, (l[0], l[1]), 1, (0, 0, 255), 1)
            box = result["bbox"]
            # 画人脸框
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # 写性别和年龄
            cv2.putText(image, '%s,%d' % (result["gender"], result["age"]), (box[0] - 1, box[1] - 4),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        return image
