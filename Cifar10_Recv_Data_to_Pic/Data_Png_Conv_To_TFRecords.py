



import tensorflow as tf
import cv2 as cv
import numpy
import os


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(path, label):
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


def bulk_load_buffer(pathlist, labelist, loadmethod):
    data_list = []
    for path in enumerate(pathlist):
        label = labelist[path[0]]
        data_list.append(loadmethod(path[1], label))
    return data_list


def get_file_pathlist(rootpath):
    pathlist = []
    for dirname, sub, filename in os.walk(rootpath):
        for name in filename:
            path = os.path.join(dirname, name)
            pathlist.append(path)
    return pathlist


def convert_to_tfr(data_list, path):
    leng = datalist.__len__()
    for data in enumerate(data_list):
        no = data[0]
        image = data[1]['raw']
        label = data[1]['label']
        rows = data[1]['row']
        cols = data[1]['col']
        depth = data[1]['depth']
        print(str(no+1)+" of "+str(leng)+" has been processed.")
        writer = tf.python_io.TFRecordWriter(path)
        image = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(int(label)),
            'image_raw': _bytes_feature(image)}))
        writer.write(example.SerializeToString())
    writer.close()





root_path = "E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\train\\cifar10_image\\airplane"

path_list = get_file_pathlist(root_path)
labelcode = 0
label_list = [int(i/i)*labelcode for i in range(1, path_list.__len__()+1)]
#method = load_image
datalist = bulk_load_buffer(path_list,label_list,load_image)
convert_to_tfr(datalist,"E:\\DownLoad\\cifar-10-python\\cifar-10-batches-py\\train\\cifar10_image\\1_one\\airplane.tfrecords")



