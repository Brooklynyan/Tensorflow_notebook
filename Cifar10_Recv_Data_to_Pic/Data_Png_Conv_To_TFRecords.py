# ==============================================================================
# Author Yan Feng
# Create Date 2017/4/23
# github.com/Brooklynyan/Tensorflow_notebook
# Programmed for Tensorflow Cifar10 dataset CNN tutorial learning
# Windows Platform
# ==============================================================================

""" Convert the cifar10 dataset from png file format to standard tfrecords file format"""


import tensorflow as tf
import cv2 as cv
import pickle as pkl
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(path, label):                                                  # load single png file return dict
    dic = {'label': None, 'row': 0, 'col': 0, 'depth': 0, 'raw': None}
    img = cv.imread(path)
    row = img.shape[0]
    col = img.shape[1]
    depth = img.shape[2]
    dic['label'] = label
    dic['row'] = row
    dic['col'] = col
    dic['depth'] = depth
    dic['raw'] = img
    return dic


def bulk_load_buffer(pathlist, labelist, loadmethod):                        # bulk load images to buffer
    data_list = []
    for path in enumerate(pathlist):
        label = labelist[path[0]]
        data_list.append(loadmethod(path[1], label))
    return data_list


def get_file_pathlist(rootpath):                                             # push png file paths into a list
    pathlist = []
    for dirname, sub, filename in os.walk(rootpath):
        for name in filename:
            path = os.path.join(dirname, name)
            pathlist.append(path)
    return pathlist


def convert_to_tfr(data_list, path):                                         # write records into single tfr file
    leng = data_list.__len__()
    writer = tf.python_io.TFRecordWriter(path)
    for data in enumerate(data_list):
        no = data[0]
        image = data[1]['raw']
        label = data[1]['label']
        rows = data[1]['row']
        cols = data[1]['col']
        depth = data[1]['depth']
        print(str(no+1)+" of "+str(leng)+" has been processed.")
        image = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(image)}))
        writer.write(example.SerializeToString())
    writer.close()


def readfile(filepath):                                               # read cpickle file and get numpy ndarray
    file = open(filepath, 'rb')
    dic = pkl.load(file, encoding='bytes')
    file.close()
    return dic


root_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\train\\cifar10_image"
meta_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\batches.meta"                   # batches.meta path
tfr_path = "E:\\Cifar10\\Train"
labelnames = list(map(lambda x: str(x).replace("b'", "", 1).replace("'", "", 1), readfile(meta_path)[b'label_names']))

for root, subpath, file_name in os.walk(root_path):
    for vsub in subpath:
        flag = vsub
        vpath = os.path.join(root, vsub)
        label_value = labelnames.index(flag)
        path_list = get_file_pathlist(vpath)
        label_list = [int(i / i) * label_value for i in range(1, path_list.__len__() + 1)]
        datalist = bulk_load_buffer(path_list, label_list, load_image)
        convert_to_tfr(datalist, tfr_path+"\\batch_"+str(label_value)+".tfrecords")


