# -------------------------------------#
#       创建YOLO类
# -------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from net.yolo3D import Yolo3DBody
import torch.backends.cudnn as cudnn
from PIL import Image, ImageFont, ImageDraw
from torch.autograd import Variable
from utils.config import Config
from utils.utils import non_max_suppression, bbox_iou, DecodeBox, distance_suppression, yolo_correct_boxes


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
# --------------------------------------------#
class YOLO(object):
    _defaults = {
        #"model_path": 'logs/Epoch575_train_loss_43.470892_val_loss_82.821692.pth',
        "classes_path": 'data/classes.txt',
        "model_image_size": (128, 128, 128, 1),
        "confidence": 0.55,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, mode, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.config = Config
        self.model_path = mode
        self.generate()

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def generate(self):
        self.config["yolo"]["classes"] = len(self.class_names)
        self.net = Yolo3DBody(self.config)

        # 加快模型训练的效率
        #print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #print('Loading weights into state dict...')
        checkpoint = torch.load(self.model_path)
        self.net.load_state_dict(checkpoint['model'])






        #state_dict = torch.load(self.model_path, map_location=device)
        #self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '1'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        self.yolo_decodes = []
        for i in range(2):
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"],
                                               (self.model_image_size[2], self.model_image_size[1], self.model_image_size[0])))

        #print('{} model, anchors, and classes loaded.'.format(self.model_path))


    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, images):
        if self.cuda:
            images = images.cuda()

        with torch.no_grad():
            outputs = self.net(images)
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                   conf_thres=self.confidence,
                                                   nms_thres=0.1)

        if (batch_detections[0]!=None):
            batch_detections = batch_detections[0].cpu().numpy()

            batch_detections = distance_suppression(batch_detections, 0.5)

            top_bboxes = np.array(batch_detections[:, :6])

            return top_bboxes
        else:
            return []

