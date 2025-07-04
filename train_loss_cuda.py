
import tifffile
import torch.optim as optim
import cv2 as cv
import numpy as np
import time
import torch
from utils.dataloader import YoloDataset, yolo_dataset_collate
import  os
from net.darknet import *
from net.yolo3D import *
from net.yolo3Dtraining import *
from torch.autograd import Variable
from utils.config import Config
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils.logger import *
import time
from utils.vessel_node_evaluate import  *

from detect import detect_image


if __name__ == "__main__":

    logger = Logger2("tensorboard")

    train_path = 'train.txt'
    with open(train_path) as f:
        lines = f.readlines()

    eval_path = 'val.txt'
    with open(eval_path) as f:
        lines_val = f.readlines()

    lr = 0.0001
    Batch_size = 1
    epochs = 1

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model = Yolo3DBody(Config)
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)


    Cuda = True
    # -------------------------------#
    #   Dataloder的使用
    # -------------------------------#
    Use_Data_Loader = True

    if True:
        print('Loading weights into state dict...')
        checkpoint = torch.load("logs/Epoch696_train_loss_6.239284515380859_val_loss_14.31109364827474.pth")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        epochs = checkpoint['epoch'] + 1

        print('Finished!')


    if Cuda:
        net = torch.nn.DataParallel(model.cuda(), device_ids=[0])
        cudnn.benchmark = False


    yolo_losses = []
    for i in range(2):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 3]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"], Config["img_d"]),
                                    Cuda))



    train_dataset = YoloDataset(lines, (Config["img_h"], Config["img_w"], Config["img_d"]),True)
    val_dataset = YoloDataset(lines_val, (Config["img_h"], Config["img_w"], Config["img_d"]),False)


    gen = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=1, pin_memory=True, drop_last=True, collate_fn = yolo_dataset_collate)
    val_gen = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=1, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

    iteration_train = 0
    iteration_val = 0

    for epoch in range(epochs, 5000):


        total_loss = 0
        val_loss = 0
        #print('start training:')

        for iteration_train, batch in enumerate(gen):

            net.train()
            batches_done = len(gen) * epoch + iteration_train

            images, targets = batch[0], batch[1]

            if Cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)


            losses = []
            for i in range(2):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])

            loss = sum(losses)


            loss.backward()
            optimizer.step()

            tensorboard_log = []
            tensorboard_log += [("train loss iter", loss.item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            tensorboard_learnrate = []
            tensorboard_learnrate += [("learning rate iter", optimizer.state_dict()['param_groups'][0]['lr'])]
            logger.list_of_scalars_summary(tensorboard_learnrate, batches_done)

            total_loss += loss
            #print("iter: %d, train loss is %f"%(iteration_train, loss.item()))

        scheduler.step()

        #print('start validation:')

        with torch.no_grad():
            for iteration_val, batch_val in enumerate(val_gen):

                net.eval()

                images_val, targets_val = batch_val[0], batch_val[1]
                if Cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    #targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor) ) for ann in targets_val]
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(2):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
                #print("iter: %d, validation loss is %f"%(iteration_val, loss.item()))

        evaluation_metrics = [
            ("train loss epoch", total_loss.item() /(iteration_train+1)),
            ("val loss epoch", val_loss.item() / (iteration_val+1)),
        ]
        logger.list_of_scalars_summary(evaluation_metrics, epoch)


        print('--------Epoch %d total train loss: %f  total val loss: %f--------' % (epoch, total_loss.item() /(iteration_train+1), val_loss.item() / (iteration_val+1)))

        if (epoch % 1 ==0):
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}

            path = 'logs/Epoch' +str(epoch)+'_train_loss_'+str(total_loss.item()/(iteration_train+1))+'_val_loss_'+ str(val_loss.item()/ (iteration_val+1))+ '.pth'
            torch.save(state, path)

            if(epoch > 1000):
                detect_image(path)
                swcs_t_path = "G:/code/vessel_3D_detect_model_data2/data/detect/model_data/truth_node_new/"
                swcs_p_path = "G:/code/vessel_3D_detect_model_data2/data/detect/model_data/predict/"
                f1, prec, rec = calculate_f1_socre(swcs_t_path, swcs_p_path)

