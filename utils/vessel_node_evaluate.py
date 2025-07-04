import numpy as np
import os
import math
import xlsxwriter

def open_swcs2numpy(file_list):
    swc_list = []
    for i in range(len(file_list)):
        data = np.loadtxt(file_list[i])

        if(data.ndim==1 & len(data)!=0):
            data = data[None]

        swc_list.append(data)
    return swc_list


def open_swc2numpy(file):
    data = np.loadtxt(file)
    return data


def distance(position1,position2):
    d = math.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2+(position1[2]-position2[2])**2)
    return d

def computing_score(swc_truth,swc_predict):
    precision = 0
    recall = 0
    f1_score =0
    acc_nodes = 0
    truth_nodes = len(swc_truth)
    predict_nodes = len(swc_predict)

    for i in range(len(swc_truth)):
        postion_t = swc_truth[i][2:5]
        r_t = swc_truth[i][5]

        for j in range(len(swc_predict)):
            postion_p = swc_predict[j][2:5]

            dis = distance(postion_t,postion_p)
            if(dis < 6):
                acc_nodes = acc_nodes+1

                swc_predict[j][2] = -100
                swc_predict[j][3] = -100
                swc_predict[j][4] = -100

                break

    precision = acc_nodes / (predict_nodes+0.000001)
    recall = acc_nodes / (truth_nodes+0.000001)
    f1_score = 2*(precision*recall)/(precision+recall+0.000001)

    return f1_score,precision,recall




def calculate_f1_socre(truth_path,predict_path):



    workbook = xlsxwriter.Workbook('VBnet_model_data.xlsx')  # 建立文件

    worksheet = workbook.add_worksheet('VBnet')  # 建立sheet，

    swcs_p_path = predict_path

    temp_swc = os.listdir(swcs_p_path)
    swcfiles_list = []
    for swc in temp_swc:
        if swc.endswith(".swc"):
            path = os.path.join(swcs_p_path, swc)
            swcfiles_list.append(path)

    swc_list_p = open_swcs2numpy(swcfiles_list)

    swcs_t_path = truth_path

    temp_swc = os.listdir(swcs_t_path)
    swcfiles_list = []
    for swc in temp_swc:
        if swc.endswith(".swc"):
            path = os.path.join(swcs_t_path, swc)
            swcfiles_list.append(path)

    swc_list_t = open_swcs2numpy(swcfiles_list)

    f1 = 0
    prec = 0
    rec = 0
    for i in range(len(swc_list_t)):
        f1_t, prec_t, rec_t = computing_score(swc_list_t[i], swc_list_p[i])

        #worksheet.write('A1', 'Hello world')  # 向A1写入

        worksheet.write(i, 1, f1_t*100)  # 向第二行第二例写入guoshun
        worksheet.write(i, 2, prec_t*100)  # 向第二行第二例写入guoshun
        worksheet.write(i, 3, rec_t*100)  # 向第二行第二例写入guoshun

        #print("No.%d f1: %f, prec: %f, rec: %f" % (i+1, f1_t, prec_t,rec_t))

        f1 = f1 + f1_t
        prec = prec + prec_t
        rec = rec + rec_t

    f1 = f1 / len(swc_list_t)
    prec = prec / len(swc_list_t)
    rec = rec / len(swc_list_t)

    print("avage f1: %f, prec: %f, rec: %f" % (f1, prec, rec))

    workbook.close()

    return f1,prec,rec






def calculate_f1_socre_save_txt(truth_path,predict_path):



    swcs_p_path = predict_path

    temp_swc = os.listdir(swcs_p_path)
    swcfiles_list = []
    for swc in temp_swc:
        if swc.endswith(".swc"):
            path = os.path.join(swcs_p_path, swc)
            swcfiles_list.append(path)

    swc_list_p = open_swcs2numpy(swcfiles_list)

    swcs_t_path = truth_path

    temp_swc = os.listdir(swcs_t_path)
    swcfiles_list = []
    for swc in temp_swc:
        if swc.endswith(".swc"):
            path = os.path.join(swcs_t_path, swc)
            swcfiles_list.append(path)

    swc_list_t = open_swcs2numpy(swcfiles_list)

    f1 = 0
    prec = 0
    rec = 0
    for i in range(len(swc_list_t)):
        f1_t, prec_t, rec_t = computing_score(swc_list_t[i], swc_list_p[i])


        print("No.%d f1: %f, prec: %f, rec: %f" % (i+1, f1_t, prec_t,rec_t))





        f1 = f1 + f1_t
        prec = prec + prec_t
        rec = rec + rec_t

    f1 = f1 / len(swc_list_t)
    prec = prec / len(swc_list_t)
    rec = rec / len(swc_list_t)

    print("avage f1: %f, prec: %f, rec: %f" % (f1, prec, rec))

    f = "lucky.txt"
    with open(f, "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
       file.write(str(prec) + "      " + str(rec) + "      " + str(f1) + "\n")
    return f1,prec,rec




if __name__ == '__main__':



    swcs_t_path = "G:/code/vessel_3D_detect_model_data/data/detect/model_data/truth_node_new/"
    swcs_p_path = "G:/code/vessel_3D_detect_model_data/data/detect/model_data/predict/"
    f1, prec, rec = calculate_f1_socre(swcs_t_path, swcs_p_path)

