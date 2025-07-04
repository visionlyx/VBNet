from random import shuffle
import numpy as np
import tifffile
from torch.utils.data.dataset import Dataset
import random
from utils.utils import *




class YoloDataset(Dataset):
    def __init__(self, train_lines, image_size, is_train):
        super(YoloDataset, self).__init__()

        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self):
        return self.train_batches


    def random_transpone(self, img, box):

        index = random.randint(0, 5)

        if (box.ndim == 1):
            box = np.expand_dims(box, 0)


        if(index == 0):
            return img, box
        if(index == 1):
            return img.transpose(0, 2, 1), box[:, [1, 0, 2, 3, 4]]
        if (index == 2):
            return img.transpose(1, 0, 2), box[:, [0, 2, 1, 3, 4]]
        if (index == 3):
            return img.transpose(1, 2, 0), box[:, [2, 0, 1, 3, 4]]
        if (index == 4):
            return img.transpose(2, 0, 1), box[:, [1, 2, 0, 3, 4]]
        if (index == 5):
            return img.transpose(2, 1, 0), box[:, [2, 1, 0, 3, 4]]

    def get_data(self, annotation_line):

        line = annotation_line.split()

        image = tifffile.imread(line[0])




        box = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]])

        if (self.is_train == True):



            #random transform
            z = random.randint(0, 360)
            y = random.randint(0, 360)
            x = random.randint(0, 360)

            matrix = getmetrix(math.radians(z), math.radians(y), math.radians(x))


            image = Rotate3Dimage(image, matrix)
            #如果是128pix
            box = RotateSWC(box, matrix, [64, 64, 64])

            image, box = self.random_transpone(image, box)

            # 如果是200PIx
            # box = RotateSWC(box, matrix, [100, 100, 100])

        else:
            z = 0
            y = 0
            x = 0

            matrix = getmetrix(math.radians(z), math.radians(y), math.radians(x))

            image = Rotate3Dimage(image, matrix)
            # 如果是128pix
            box = RotateSWC(box, matrix, [64, 64, 64])

            # 如果是200PIx
            # box = RotateSWC(box, matrix, [100, 100, 100])

        #旋转的中心坐标（图像中心点）
        #如果是200PIx
        #box = RotateSWC(box, matrix, [100, 100, 100])



        #输出tif swc到本地
        '''
        test_tif_out ='G:/code/vessel_3D_detect_model_data/data/temp_check/check.tif'
        test_swc_out = 'G:/code/vessel_3D_detect_model_data/data/temp_check/check.swc'
        tifffile.imwrite(test_tif_out, image)

        fp = open(test_swc_out, 'w')
        for ik in range(len(box)):
            fp.write(str(ik + 1))  # id
            fp.write(" ")
            fp.write(str(int(2)))  # type
            fp.write(" ")
            fp.write(str((box[ik][0])))  # x
            fp.write(" ")
            fp.write(str((box[ik][1])))  # y
            fp.write(" ")
            fp.write(str((box[ik][2])))  # z
            fp.write(" ")
            fp.write(str((box[ik][3])))
            fp.write(" ")
            fp.write(str(-1))
            fp.write('\n')
        fp.close()
        '''


        if len(box) == 0:
            print('null')
            return image, []
        else:
            return image, box

    def __getitem__(self, index):

        #t1 = time.clock()

        if index == 0:
            shuffle(self.train_lines)
        lines = self.train_lines
        n = self.train_batches
        index = index % n

        img, y = self.get_data(lines[index])

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            # x_center y_center z_center length
            target = np.array(y[:, :4], dtype=np.float32)

            target[:, 0] = target[:, 0] / self.image_size[2]

            target[:, 1] = target[:, 1] / self.image_size[1]

            target[:, 2] = target[:, 2] / self.image_size[0]

            target[:, 3] = target[:, 3] / self.image_size[2]

            target = np.maximum(np.minimum(target, 1), 0)

            y = np.concatenate([target, y[:, -1:]], axis=-1)

        tmp_inp = np.transpose((img-img.min()) / (img.max()-img.min()), (0, 1, 2))   # z y x

        tmp_inp = tmp_inp[np.newaxis, :]    # c z y x

        tmp_inp = np.array(tmp_inp, dtype=np.float32)

        tmp_targets = np.array(y, dtype=np.float32)
        #t2 = time.clock()
        #print("loaddddd used:", t2-t1)

        return tmp_inp, tmp_targets

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    #bboxes = np.array(bboxes)
    return images, bboxes


