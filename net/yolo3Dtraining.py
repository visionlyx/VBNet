import cv2
from random import shuffle
import numpy as np
import torch
import torch.nn as nn
import math
from PIL import Image
from utils.utils import bbox_iou


def jaccard(_box_a, _box_b):
    b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 3] / 2, _box_a[:, 0] + _box_a[:, 3] / 2
    b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
    b1_z1, b1_z2 = _box_a[:, 2] - _box_a[:, 3] / 2, _box_a[:, 2] + _box_a[:, 3] / 2

    b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 3] / 2, _box_b[:, 0] + _box_b[:, 3] / 2
    b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2
    b2_z1, b2_z2 = _box_b[:, 2] - _box_b[:, 3] / 2, _box_b[:, 2] + _box_b[:, 3] / 2

    #print(_box_a.shape[0])
    #print(_box_b.shape[0])

    box_a = torch.zeros(int(_box_a.shape[0]), 6)
    box_b = torch.zeros(int(_box_b.shape[0]), 6)


    box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3], box_a[:, 4], box_a[:, 5] = b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2
    box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3], box_b[:, 4], box_b[:, 5] = b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2
    A = box_a.size(0)
    B = box_b.size(0)


    max_xyz = torch.min(box_a[:, 3:].unsqueeze(1).expand(A, B, 3),
                       box_b[:, 3:].unsqueeze(0).expand(A, B, 3))
    min_xyz = torch.max(box_a[:, :3].unsqueeze(1).expand(A, B, 3),
                       box_b[:, :3].unsqueeze(0).expand(A, B, 3))
    inter = torch.clamp((max_xyz - min_xyz), min=0)

    #print(torch.max(max_xyz))
    #print(torch.max(min_xyz))



    inter = inter[:, :, 0] * inter[:, :, 1] * inter [:, :, 2]

    #print(torch.max(inter))
    # 计算先验框和真实框各自的面积
    area_a = ((box_a[:, 3] - box_a[:, 0]) *
              (box_a[:, 4] - box_a[:, 1]) * (box_a[:, 5] - box_a[:, 2])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 3] - box_b[:, 0]) *
              (box_b[:, 4] - box_b[:, 1]) * (box_b[:, 5] - box_b[:, 2])).unsqueeze(0).expand_as(inter)  # [A,B]
    # 求IOU
    union = area_a + area_b - inter

    #print(torch.max(area_a))
    #print(torch.max(area_b))
    #print(torch.max(union))

    iou = inter / union

    #print(torch.max(iou))
    #print(torch.min(iou))



    return inter / union  # [A,B]


def clip_by_tensor(t, t_min, t_max):
    t = t.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


def MSELoss(pred, target):
    return (pred - target) ** 2


def BCELoss(pred, target):
    epsilon = 1e-7
    pred = clip_by_tensor(pred, epsilon, 1.0 - epsilon)
    output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
    return output


class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size, cuda):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.feature_length = [img_size[0] // 16, img_size[0] // 8]
        self.img_size = img_size

        self.ignore_threshold = 0.5
        self.lambda_xyz = 1
        self.lambda_length = 1.0
        self.lambda_conf = 1
        self.lambda_cls = 1.0
        self.cuda = cuda

    def forward(self, input, targets=None):
        # input为bs,3*(5+num_classes), depth, height, weight

        #input: [bs, 18, 8, 8, 8]  or [bs, 18, 16, 16, 16]




        # 一共多少张图片
        bs = input.size(0)
        # 特征层的Z
        in_d = input.size(2)
        # 特征层的Y
        in_h = input.size(3)
        # 特征层的X
        in_w = input.size(4)



        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        stride_d = self.img_size[2] / in_d
        stride_h = self.img_size[1] / in_h
        stride_w = self.img_size[0] / in_w

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(a_w / stride_w, a_h / stride_h , a_d / stride_d) for a_w, a_h, a_d in self.anchors]

        # bs,3*(5+num_classes),8,8,8 -> bs,3,8,8,8(5+num_classes)
        #


        ##调整input的尺寸
        prediction = input.view(bs, int(self.num_anchors / 2),
                                self.bbox_attrs, in_d, in_h, in_w).permute(0, 1, 3, 4, 5, 2).contiguous()


        # prediciton 大小为 [bs, 3, 8, 8, 8, 6]  其中6 [x, y, z, l, conf, cls]


        # 对prediction预测进行调整


        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        z = torch.sigmoid(prediction[..., 2])  # Center z

        l = prediction[..., 3]  # length
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        #print(torch.max(conf))

        # 找到哪些先验框内部包含物体
        mask, noobj_mask, tx, ty, tz, tl, tconf, tcls, box_loss_scale_x, box_loss_scale_y, box_loss_scale_z = \
            self.get_target(targets, scaled_anchors,
                            in_w, in_h, in_d,
                            self.ignore_threshold)

        #noobj_mask = self.get_ignore(prediction, targets, scaled_anchors, in_w, in_h, in_d, noobj_mask)
        if self.cuda:
            box_loss_scale_x = (box_loss_scale_x).cuda()
            box_loss_scale_y = (box_loss_scale_y).cuda()
            box_loss_scale_z = (box_loss_scale_z).cuda()

            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()

            tx, ty, tz, tl = tx.cuda(), ty.cuda(), tz.cuda(), tl.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()

        #box_loss_scale = 2 - box_loss_scale_x * box_loss_scale_y * box_loss_scale_z

        #  losses.
        #loss_x = torch.sum(BCELoss(x, tx) / bs * box_loss_scale * mask)
        #loss_y = torch.sum(BCELoss(y, ty) / bs * box_loss_scale * mask)
        #loss_z = torch.sum(BCELoss(z, tz) / bs * box_loss_scale * mask)

        #loss_l = torch.sum(MSELoss(l, tl) / bs * 0.5 * box_loss_scale * mask)

        loss_x = torch.sum(MSELoss(x, tx) / bs * mask)
        loss_y = torch.sum(MSELoss(y, ty) / bs * mask)
        loss_z = torch.sum(MSELoss(z, tz) / bs * mask)

        loss_l = torch.sum(MSELoss(l, tl) / bs * mask)


        #bce_loss(pred_conf[obj_mask], tconf[obj_mask])
        #onf1 = torch.sum(BCELoss(conf[mask], tconf[mask]) / bs)
        #conf2 = torch.sum(BCELoss(conf[noobj_mask], tconf[noobj_mask]) / bs)

        conf1 = torch.sum(BCELoss(conf, tconf) * mask / bs)
        conf2 = torch.sum(BCELoss(conf, tconf) * noobj_mask / bs) *0.015

        #修改之前的conf
        #conf1 = torch.sum(BCELoss(conf, mask) * mask / bs)
        #conf2 = torch.sum(BCELoss(conf, mask) * noobj_mask / bs)
        loss_conf = conf1 + conf2
        #loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + \
        #            torch.sum(BCELoss(conf, mask) * noobj_mask / bs)

        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]) / bs)

        loss = loss_x * self.lambda_xyz + loss_y * self.lambda_xyz + \
               loss_z * self.lambda_xyz + loss_l * self.lambda_length + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
        # print(loss, loss_x.item() + loss_y.item(), loss_w.item() + loss_h.item(),
        #         loss_conf.item(), loss_cls.item(), \
        #         torch.sum(mask),torch.sum(noobj_mask))

        #print("lossx: %f, lossy: %f, lossz: %f, loss_l: %f, loss_conf: %f   "%(loss_x.item(),loss_y.item(),loss_z.item(),loss_l.item(),loss_conf.item()))


        return loss, loss_x.item(), loss_y.item(), loss_z.item(), \
               loss_l.item(), loss_conf.item(), loss_cls.item()

    def get_target(self, target, anchors, in_w, in_h, in_d, ignore_threshold):
        # 计算一共有多少张图片
        bs = len(target)

        #target [bs, n, 5]  5:[x,y,z,l,n]

        # 获得先验框
        anchor_index = [[0, 1, 2], [3, 4, 5]][self.feature_length.index(in_w)]

        subtract_index = [0, 3][self.feature_length.index(in_w)]

        # 创建全是0或者全是1的阵列
        mask = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        noobj_mask = torch.ones(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        tz = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        tl = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, self.num_classes, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)
        box_loss_scale_z = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False)


        for b in range(bs):
            for t in range(target[b].shape[0]):
                # 计算出在特征层上的点位
                gx = target[b][t, 0] * in_w
                gy = target[b][t, 1] * in_h
                gz = target[b][t, 2] * in_d
                gl = target[b][t, 3] * in_w  #length 各向同性in_w = in_h = in_d

                # 计算出属于哪个网格
                gi = int(gx)
                gj = int(gy)
                gk = int(gz)

                gw = (gx+gl)-(gx-gl)
                gh = (gy+gl)-(gy-gl)
                gd = (gz+gl)-(gz-gl)




                # 计算真实框的位置
                #gt_box = torch.FloatTensor(np.array([0, 0, 0, gw, gh, gd])).unsqueeze(0)
                gt_box = torch.FloatTensor([0, 0, 0, gw, gh, gd]).unsqueeze(0)
                # 计算出所有先验框的位置
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((self.num_anchors, 3)),
                                                                  np.array(anchors)), 1))
                # 计算重合程度  
                anch_ious = bbox_iou(gt_box, anchor_shapes)

                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)

                anch_ious_current = anch_ious[subtract_index:subtract_index + 3]
                noobj_mask[b, anch_ious_current > 0.5, gk, gj, gi] = 0


                if best_n not in anchor_index:
                    continue
                # Masks
                if (gj < in_h) and (gi < in_w) and (gk < in_d):
                    best_n = best_n - subtract_index
                    # 判定哪些先验框内部真实的存在物体
                    noobj_mask[b, best_n, gk, gj, gi] = 0
                    #修改成带权重的mask，和半径相关
                    #mask[b, best_n, gk, gj, gi] = 1
                    mask[b, best_n, gk, gj, gi] = torch.pow((gl/in_w*self.img_size[0] - 5), 2) / 5 + 1

                    #mask[b, best_n, gk, gj, gi] = torch.pow((gl-10/(self.img_size[0]/in_w)),2)/(10/(self.img_size[0]/in_w)) + 1

                    #anch_ious_current = anch_ious[subtract_index:subtract_index+3]
                    #noobj_mask[b, anch_ious_current > 0.5, gk, gj, gi] = 0

                    # 计算先验框中心调整参数
                    tx[b, best_n, gk, gj, gi] = gx - gi
                    ty[b, best_n, gk, gj, gi] = gy - gj
                    tz[b, best_n, gk, gj, gi] = gz - gk

                    # 计算先验框宽高调整参数
                    #print(gw, "  ", anchors[best_n + subtract_index][0])
                    tl[b, best_n, gk, gj, gi] = math.log((gw+gh+gd) / 3 / anchors[best_n + subtract_index][0])
                    #print(gw, "  ",anchors[best_n + subtract_index][0])


                    # 用于获得xywh的比例
                    box_loss_scale_x[b, best_n, gk, gj, gi] = target[b][t, 3] * 2
                    box_loss_scale_y[b, best_n, gk, gj, gi] = target[b][t, 3] * 2
                    box_loss_scale_z[b, best_n, gk, gj, gi] = target[b][t, 3] * 2


                    # 物体置信度
                    tconf[b, best_n, gk, gj, gi] = 1
                    # 种类
                    tcls[b, best_n, gk, gj, gi, int(target[b][t, 4])] = 1
                else:
                    print('Step {0} out of bound'.format(b))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue

        return mask, noobj_mask, tx, ty, tz, tl, tconf, tcls, box_loss_scale_x, box_loss_scale_y, box_loss_scale_z

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, in_d, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        # print(scaled_anchors)
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        z = torch.sigmoid(prediction[..., 2])
        # 先验框的宽高调整参数
        l = prediction[..., 3]  # Width

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor


        grid_x = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False).type(FloatTensor)
        grid_y = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False).type(FloatTensor)
        grid_z = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False).type(FloatTensor)
        for b_index in range(bs):
            for anc_index in range(int(self.num_anchors / 2)):
                for z_index in range(int(in_d)):
                    for y_index in range(int(in_h)):
                        for x_index in range(int(in_w)):
                            grid_x[b_index][anc_index][z_index][y_index][x_index] = x_index
                            grid_y[b_index][anc_index][z_index][y_index][x_index] = y_index
                            grid_z[b_index][anc_index][z_index][y_index][x_index] = z_index


        #FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        #LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        #grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
        #    int(bs * self.num_anchors / 3), 1, 1).view(x.shape).type(FloatTensor)
        #grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
        #    int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)





        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_d = FloatTensor(scaled_anchors).index_select(1, LongTensor([2]))


        anchor_lenght = torch.zeros(bs, int(self.num_anchors / 2), in_d, in_h, in_w, requires_grad=False).cuda()
        for b_index in range(bs):
            for anc_index in range(int(self.num_anchors / 2)):
                for z_index in  range(int(in_d)):
                    for y_index in range(int(in_h)):
                        for x_index in range(int(in_w)):
                            anchor_lenght[b_index][anc_index][z_index][y_index][x_index] = anchor_w[anc_index][0]




        #anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        #anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与半径
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = z.data + grid_z
        pred_boxes[..., 3] = torch.exp(l.data) * anchor_lenght


        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            if len(target[i]) > 0:
                gx = target[i][:, 0:1] * in_w
                gy = target[i][:, 1:2] * in_h
                gz = target[i][:, 2:3] * in_d
                gl = target[i][:, 3:4] * in_w

                gw = gl * 2
                gh = gl * 2
                gd = gl * 2

                #gt_box = torch.FloatTensor(np.concatenate([gx, gy, gz, gl], -1)).type(FloatTensor)
                gt_box = torch.cat((gx,gy,gz,gl), dim=1)
                anch_ious = jaccard(gt_box, pred_boxes_for_ignore)


                for t in range(target[i].shape[0]):
                    anch_iou = anch_ious[t].view(pred_boxes[i].size()[:4])
                    noobj_mask[i][anch_iou > self.ignore_threshold] = 0

                #print(torch.max(anch_ious))
        return noobj_mask


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


class Generator(object):
    def __init__(self, batch_size,
                 train_lines, image_size,
                 ):

        self.batch_size = batch_size
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size

    def get_random_data(self, annotation_line, input_shape, jitter=.1, hue=.1, sat=1.3, val=1.3):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # resize image
        new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
        scale = rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # correct boxes
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self, train=True):
        while True:
            shuffle(self.train_lines)
            lines = self.train_lines
            inputs = []
            targets = []
            for annotation_line in lines:
                img, y = self.get_random_data(annotation_line, self.image_size[0:2])

                if len(y) != 0:
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    boxes[:, 0] = boxes[:, 0] / self.image_size[1]
                    boxes[:, 1] = boxes[:, 1] / self.image_size[0]
                    boxes[:, 2] = boxes[:, 2] / self.image_size[1]
                    boxes[:, 3] = boxes[:, 3] / self.image_size[0]

                    boxes = np.maximum(np.minimum(boxes, 1), 0)
                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                    boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
                    boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
                    y = np.concatenate([boxes, y[:, -1:]], axis=-1)
                img = np.array(img, dtype=np.float32)

                inputs.append(np.transpose(img / 255.0, (2, 0, 1)))
                targets.append(np.array(y, dtype=np.float32))
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield tmp_inp, tmp_targets

