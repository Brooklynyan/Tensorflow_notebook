# ==============================================================================
# Author Yan Feng
# Create Date 2017/4/21
# github.com/Brooklynyan/Tensorflow_notebook
# Programmed for Tensorflow Cifar10 dataset CNN tutorial learning
# Windows Platform
# ==============================================================================

""" Recover the cifar10 dataset to png file format from original cpickle files"""

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


def generate_images(rootpath, batchno, data_array, label_array, labelname_array):      # generates png files
    for image in enumerate(data_array):
        i = image[0]
        img = image[1]
        label = label_array[i]
        labelname = labelname_array[label]
        path = rootpath + "\\cifar10_image\\" + labelname + "\\" + str(batchno) + "_" + str(i) + ".png"
        cv.imwrite(path, get_image_pixel_array(img))


data_path = {"E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\data_batch_1",      # train data batch file path
             "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\data_batch_2",
             "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\data_batch_3",
             "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\data_batch_4",
             "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\data_batch_5"}

meta_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\batches.meta"        # batches.meta path
root_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\train"                # train image root folder path
labelnames = list(map(lambda x: str(x).replace("b'", "", 1).replace("'", "", 1), readfile(meta_path)[b'label_names']))

create_image_folder(root_path, labelnames)                                            # create train sub folders

for v_path in data_path:                                                             # generate train images
    batch_data = readfile(v_path)[b'data']
    batch_data_labels = readfile(v_path)[b'labels']
    batch_no = v_path[-1:]
    generate_images(root_path, batch_no, batch_data, batch_data_labels, labelnames)

data_path = {"E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\test_batch"}        # test data batch file path
root_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\test"                # test image root folder path

create_image_folder(root_path, labelnames)                                            # create test sub folders

for v_path in data_path:                                                             # generate test images
    batch_data = readfile(v_path)[b'data']
    batch_data_labels = readfile(v_path)[b'labels']
    batch_no = 1
    generate_images(root_path, batch_no, batch_data, batch_data_labels, labelnames)

