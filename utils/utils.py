from __future__ import division
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numba import jit

from PIL import Image, ImageDraw, ImageFont


class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        batch_size = input.size(0)
        input_depth = input.size(2)
        input_height = input.size(3)
        input_width = input.size(4)

        # 计算步长
        stride_d = self.img_size[2] / input_depth
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        # 归一到特征层上
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h, anchor_depth / stride_d ) for anchor_width, anchor_height, anchor_depth in
                          self.anchors]

        # 对预测结果进行resize
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_depth, input_height, input_width).permute(0, 1, 3, 4, 5, 2).contiguous()

        # prediction : bs , num_anch, 6, 8, 8, 8

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        z = torch.sigmoid(prediction[..., 2])

        l = prediction[..., 3]  # length

        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        grid_x = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()
        grid_y = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()
        grid_z = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()
        for b_index in range(batch_size):
            for anc_index in range(int(self.num_anchors)):
                for z_index in range(int(input_depth)):
                    for y_index in range(int(input_height)):
                        for x_index in range(int(input_width)):
                            grid_x[b_index][anc_index][z_index][y_index][x_index] = x_index
                            grid_y[b_index][anc_index][z_index][y_index][x_index] = y_index
                            grid_z[b_index][anc_index][z_index][y_index][x_index] = z_index



        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))


        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_d = FloatTensor(scaled_anchors).index_select(1, LongTensor([2]))

        anchor_lenght = torch.zeros(batch_size, int(self.num_anchors), input_depth, input_height, input_width, requires_grad=False).cuda()
        #anchor_lenght = LongTensor(anchor_lenght)


        for b_index in range(batch_size):
            for anc_index in range(int(self.num_anchors)):
                for z_index in range(int(input_depth)):
                    for y_index in range(int(input_height)):
                        for x_index in range(int(input_width)):
                            anchor_lenght[b_index][anc_index][z_index][y_index][x_index] = anchor_w[anc_index][0]

        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = z.data + grid_z
        pred_boxes[..., 3] = torch.exp(l.data) * anchor_lenght

        lll = pred_boxes[..., 3].view(-1)
        #print(torch.max(lll))
        #print(torch.min(lll))
        # 用于将输出调整为相对于416x416的大小
        _scale = torch.Tensor([stride_w, stride_h, stride_d, stride_w]).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data




def getmetrix(zbeta, ybeta, xbeta):  # 旋转
    # beta>0表示逆时针旋转；beta<0表示顺时针旋转
    transformZ = np.array([[math.cos(zbeta), -math.sin(zbeta), 0],
                                [math.sin(zbeta), math.cos(zbeta), 0],
                                [0, 0, 1]])
    transformY = np.array([[math.cos(ybeta), 0, math.sin(ybeta)],
                                [0, 1, 0],
                                [-math.sin(ybeta), 0, math.cos(ybeta)]])

    transformX = np.array([[1, 0, 0],
                                [0, math.cos(xbeta), -math.sin(xbeta)],
                                [0, math.sin(xbeta), math.cos(xbeta)]])

    temp2 = np.dot(transformZ, transformY)
    transformZYX = np.dot(temp2, transformX)
    return transformZYX





@jit(nopython=True)
def Rotate3Dimage(src_image, metrix):
    dst_image =np.zeros((src_image.shape[0], src_image.shape[1],src_image.shape[2]),dtype=np.uint8)
    for k in range(src_image.shape[0]):
        for j in range(src_image.shape[1]):
            for i in range(src_image.shape[2]):
                src_pos=np.array([i-src_image.shape[2]/2, j-src_image.shape[1]/2, k-src_image.shape[0]/2])

                [x, y, z] = np.dot(metrix, src_pos)

                x = int(x) + int(src_image.shape[2]/2)
                y = int(y) + int(src_image.shape[1]/2)
                z = int(z) + int(src_image.shape[0]/2)

                if x >= src_image.shape[2] or y >= src_image.shape[1] or z >= src_image.shape[2] or x < 0 or y < 0 or z < 0:
                    dst_image[k][j][i] = 0
                else:
                    dst_image[k][j][i] = src_image[z][y][x]
    return dst_image[0:128, 0:128, 0:128]
    #------------------------------------------------------------------------------------------------------
    #return dst_image[36:164,36:164,36:164]




def RotateSWC(swcfile,  metrix, center):
    newswc = list()
    bound_pix = 5
    #去除边界效应
    for i in range(len(swcfile)):
        current_node = swcfile[i].copy()
        src_swc_node = np.array([swcfile[i][0] - center[0], swcfile[i][1] - center[1], swcfile[i][2] - center[2]])
        temp = np.linalg.inv(metrix)

        [x, y, z] = np.dot(temp, src_swc_node)

        x = x + center[0]
        y = y + center[1]
        z = z + center[2]

        #if x >= center[0]*2 or y >= center[1]*2 or z >= center[2]*2 or x < 0 or y < 0 or z < 0:
        if x >= center[0] + 64 - bound_pix or y >= center[1] + 64 - bound_pix or z >= center[2] + 64 -bound_pix or x < center[0] - 64 +bound_pix  or y < center[1] - 64 +bound_pix  or z < center[2] - 64 +bound_pix :
            continue
        else:
            #offset
            #如果是200的尺寸，中心点是100，前后各减去64 偏移应该从36开始
            # ------------------------------------------------------------------------------------------------------
            #current_node[0] = x - 36
            #current_node[1] = y - 36
            #current_node[2] = z - 36
            current_node[0] = x
            current_node[1] = y
            current_node[2] = z
            newswc.append(current_node)

    newswc =np.array(newswc)
    return newswc



def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate((bottom - top, right - left), axis=-1) / input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    #print(np.shape(boxes))
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def bbox_iou(box1, box2, x1y1z1x2y2z2=True):
    """
        计算IOU
    """
    if not x1y1z1x2y2z2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 3] / 2, box1[:, 0] + box1[:, 3] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 4] / 2, box1[:, 1] + box1[:, 4] / 2
        b1_z1, b1_z2 = box1[:, 2] - box1[:, 5] / 2, box1[:, 2] + box1[:, 5] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 3] / 2, box2[:, 0] + box2[:, 3] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 4] / 2, box2[:, 1] + box2[:, 4] / 2
        b2_z1, b2_z2 = box2[:, 2] - box2[:, 5] / 2, box2[:, 2] + box2[:, 5] / 2
    else:
        b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3], box1[:, 4], box1[:, 5]
        b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3], box2[:, 4], box2[:, 5]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_z1 = torch.max(b1_z1, b2_z1)

    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_rect_z2 = torch.min(b1_z2, b2_z2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0) * \
                 torch.clamp(inter_rect_z2 - inter_rect_z1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1) * (b1_z2 - b1_z1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) * (b2_z2 - b2_z1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def non_max_suppression(prediction, num_classes, conf_thres, nms_thres):
    # 求左上角和右下角



    box_corner = prediction.new(prediction.shape)

    x1 = prediction[:, :, 0:1] - prediction[:, :, 3:4] / 2
    y1 = prediction[:, :, 1:2] - prediction[:, :, 3:4] / 2
    z1 = prediction[:, :, 2:3] - prediction[:, :, 3:4] / 2
    x2 = prediction[:, :, 0:1] + prediction[:, :, 3:4] / 2
    y2 = prediction[:, :, 1:2] + prediction[:, :, 3:4] / 2
    z2 = prediction[:, :, 2:3] + prediction[:, :, 3:4] / 2

    prediction = torch.cat((x1, y1, z1, x2, y2, z2, prediction[:, :, 4:]), -1)

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 利用置信度进行第一轮筛选
        conf_mask = (image_pred[:, 6] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # 获得种类及其置信度
        class_conf, class_pred = torch.max(image_pred[:, 7:7 + num_classes], 1, keepdim=True)

        # 获得的内容为(x1, y1, z1,  x2, y2, z2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :7], class_conf.float(), class_pred.float()), 1)

        # 获得种类
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()

        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]
            # 按照存在物体的置信度排序
            _, conf_sort_index = torch.sort(detections_class[:, 6], descending=True)
            detections_class = detections_class[conf_sort_index]
            # 进行非极大抑制
            max_detections = []
            while detections_class.size(0):
                # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                max_detections.append(detections_class[0].unsqueeze(0))
                if len(detections_class) == 1:
                    break
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                detections_class = detections_class[1:][ious < nms_thres]
            # 堆叠
            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output

def distance(position1,position2):
    d = math.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2+(position1[2]-position2[2])**2)
    return d

def distance_suppression(prediction,dis_thres):
    output = []




    x_c = (prediction[:,3] + prediction[:,0])/2
    y_c = (prediction[:,4] + prediction[:,1])/2
    z_c = (prediction[:, 5] + prediction[:, 2])/2

    #position = torch.cat((x_c,y_c,z_c)

    r = (prediction[:,3] - prediction[:,0]) / 2
    prediction_copy = prediction.tolist()


    for i in range(len(prediction)):
        temp = []
        temp.append(prediction[i])
        for j in range(len(prediction)):
            if i==j:
                continue
        #x1, y1, z1, x2, y2, z2
            d = math.sqrt((x_c[i]-x_c[j])**2+(y_c[i]-y_c[j])**2+(z_c[i]-z_c[j])**2)
            if d < (r[i]+r[j]) * dis_thres or d < 1:
                temp.append(prediction[j])

        best_conf = temp[0]
        if(len(temp)>1):
            for k in range (1,len(temp)):
                if temp[k][6]>best_conf[6]:
                    best_conf = temp[k]
        output.append(best_conf)


    #output = list(set(output))

    resultList = []
    resultList.append(output[0])
    for i in range(1,len(output)):

        copy = 0
        for tt in resultList:
            if (tt == output[i]).all():
                copy =1

        if copy==0:
            resultList.append(output[i])

    resultList = np.array(resultList)
    return resultList


