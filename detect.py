from yolovessel import YOLO
import numpy as np
import tifffile
import torch
import os
from torch.autograd import Variable

from utils.vessel_node_evaluate import  *



def detect_image(model_path):
    yolo = YOLO(model_path)
    test_file_name = 'G:/code/vessel_3D_detect_model_data2/data/datasets/test.txt'

    tiff_path = 'G:/code/vessel_3D_detect_model_data2/data/detect/model_data/image/'
    dst = 'G:/code/vessel_3D_detect_model_data2/data/detect/model_data/predict/'

    image_ids = open(test_file_name).read().strip().split()

    for image_id in image_ids:
        image_file = image_id + '.tif'
        temp_tif = os.path.join(tiff_path, image_file)

        img = tifffile.imread(temp_tif)

        img = img[np.newaxis, :]

        img = np.array(img, dtype=np.float32)

        #img = img / 255

        img = (img - img.min()) / (img.max() - img.min())

        images = Variable(torch.from_numpy(img).type(torch.FloatTensor))
        images = images.unsqueeze(0)

        r_image = yolo.detect_image(images)
        out_file_name = image_id + '_pre.swc'
        temp_swc = os.path.join(dst, out_file_name)
        fp = open(temp_swc, 'w')
        for i in range(len(r_image)):
            fp.write(str(i + 1))
            fp.write(" ")
            fp.write('2')
            fp.write(" ")
            fp.write(str((int(r_image[i][3]) + (int(r_image[i][0]))) / 2))
            fp.write(" ")
            fp.write(str((int(r_image[i][4]) + (int(r_image[i][1]))) / 2))
            fp.write(" ")
            fp.write(str((int(r_image[i][5]) + (int(r_image[i][2]))) / 2))
            fp.write(" ")
            fp.write(str((int(r_image[i][3]) - (int(r_image[i][0]))) / 2 / 3))
            fp.write(" ")
            fp.write(str(-1))
            fp.write('\n')
        fp.close()




def detect_image_dir_image(src_dir,dst_dir,model_path):
    yolo = YOLO(model_path)

    tiff_path = src_dir
    dst =  dst_dir

    list_file = os.listdir(tiff_path)

    for i in range(0, len(list_file)):
        img = tifffile.imread(os.path.join(tiff_path, list_file[i]))
        img = img[np.newaxis, :]

        img = np.array(img, dtype=np.float32)


        if img.max() == 0:
            out_file_name = list_file[i].split('.')[0] + '_pre.swc'
            temp_swc = os.path.join(dst, out_file_name)
            fp = open(temp_swc, 'w')

            fp.write('\n')
            fp.close()
            continue

        img = (img - img.min()) / (img.max() - img.min())

        images = Variable(torch.from_numpy(img).type(torch.FloatTensor))
        images = images.unsqueeze(0)

        r_image = yolo.detect_image(images)
        out_file_name = list_file[i].split('.')[0] + '_pre.swc'
        temp_swc = os.path.join(dst, out_file_name)
        fp = open(temp_swc, 'w')
        for i in range(len(r_image)):
            fp.write(str(i + 1))
            fp.write(" ")
            fp.write('2')
            fp.write(" ")
            fp.write(str((int(r_image[i][3]) + (int(r_image[i][0]))) / 2))
            fp.write(" ")
            fp.write(str((int(r_image[i][4]) + (int(r_image[i][1]))) / 2))
            fp.write(" ")
            fp.write(str((int(r_image[i][5]) + (int(r_image[i][2]))) / 2))
            fp.write(" ")
            fp.write(str((int(r_image[i][3]) - (int(r_image[i][0]))) / 2 / 3))
            fp.write(" ")
            fp.write(str(-1))
            fp.write('\n')
        fp.close()





if __name__ == '__main__':

    model_path = 'logs/Epoch696_train_loss_6.239284515380859_val_loss_14.31109364827474.pth'

    src = 'data/detect/model_data/image'
    dst = 'data/detect/model_data/predict/'

    detect_image_dir_image(src, dst, model_path)
    swcs_p_path = 'data/detect/model_data/predict/'
    swcs_t_path = 'data/detect/model_data/truth_node_new/'
    f1, prec, rec = calculate_f1_socre(swcs_t_path, swcs_p_path)


