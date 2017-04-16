

import os
import numpy as np
import pickle as pkl
import cv2 as cv


def create_image_folder(path, array):                               # create sub folders for each class of images
    root = path+"\\cifar10_image"
    os.mkdir(root)
    for dir_name in array:
        os.mkdir(root+"\\"+dir_name)


def readfile(filepath):                                               # read cpickle file and get numpy ndarray
    file = open(filepath, 'rb')
    dic = pkl.load(file, encoding='bytes')
    file.close()
    return dic


def get_image_pixel_array(array):                                  # return the pixel ndarray of image
    varray = np.array(array).reshape(3, 1024)
    varray = np.dstack((varray[0], varray[1], varray[2]))
    varray = varray.reshape(32, 32, 3)
    return varray


data_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\data_batch_1"
meta_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\batches.meta"
root_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\test"

labelnames = list(map(lambda x: str(x).replace("b'", "", 1).replace("'", "", 1), readfile(meta_path)[b'label_names']))
batch_data = readfile(data_path)[b'data']
batch_data_labels = readfile(data_path)[b'labels']
batch_no = data_path[-1:]

create_image_folder(root_path, labelnames)

for image in enumerate(batch_data):
    i = image[0]
    img = image[1]
    label = batch_data_labels[i]
    labelname = labelnames[label]
    path = root_path+"\\cifar10_image\\"+labelname+"\\"+str(batch_no)+"_"+str(i)+".png"
    cv.imwrite(path, get_image_pixel_array(img))

#print(readfile(data_path)[b'labels'])
#print(get_image_pixel_array(readfile(data_path)[b'data'][1]))
#print(np.array(readfile(data_path)[b'data'][1]).__len__())
#print(readfile(data_path)[b'batch_label'])
#print(readfile(data_path).keys())
#print(readfile(meta_path)[b'label_names'])

#cv.imwrite("D:\\test.png",get_image_pixel_array(readfile(file_path)[b'data'][1]))