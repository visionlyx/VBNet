import numpy as np

datasets=[ 'train',  'val',  'test']

classes = ["node"]

r_factor = 3


def convert_swc(image_id,list_file):
    in_file = open('data/swc/%s.swc'%image_id)
    data = np.loadtxt(in_file)
    node_list = list()

    list_file.write('data/image/%s.tif' % image_id)



    #x_center = data[:,2]
    #y_center = data[:,3]
    #z_center = data[:,4]
    #r = data[:,5]

    #x_min = x_center - r*r_factor
    #y_min = y_center - r * r_factor
    #z_min = z_center - r * r_factor
    #x_max = x_center + r*r_factor
    #y_max = y_center + r * r_factor
    #z_max = z_center + r * r_factor
    #data_new_ = np.vstack((x_min, y_min, z_min, x_max, y_max, z_max))
    #data_new = data_new_.transpose(1,0)

    #for i in range(data_new.shape[0]):
    #    list_file.write(" " +str(data_new[i][0])+ ","+str(data_new[i][1])+ ","+str(data_new[i][2])+ ","+str(data_new[i][3])+ ","+str(data_new[i][4])+ ","+str(data_new[i][5])+',' + str(id))
    #list_file.write('\n')



    for i in range(len(data)):
        node_list.append(data[i])

    for index in range(len(node_list)):
        id = 0
        x_center = node_list[index][2]
        y_center = node_list[index][3]
        z_center = node_list[index][4]
        r = node_list[index][5] * r_factor
        list_file.write(" " +str(x_center)+ ","+str(y_center)+ ","+str(z_center)+ ","+str(r)+ ","+ str(id))
    list_file.write('\n')







for image_set in datasets:
    image_ids = open('data/datasets/%s.txt' % image_set).read().strip().split()
    list_file = open('%s.txt' % image_set, 'w')
    for image_id in image_ids:
        convert_swc(image_id,list_file)
    list_file.close()





