# ==============================================================================
# Author Yan Feng
# Create Date 2017/4/24
# github.com/Brooklynyan/Tensorflow_notebook
# Programmed for Tensorflow Cifar10 dataset CNN tutorial learning
# Windows Platform
# ==============================================================================

""" Validate the cifar10 dataset from tfrecords file """


import tensorflow as tf
import numpy as np


def tfr_recover_data(filename):                                          # recover the serialized data from tfr file
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),              # tf.train.Example Features
               'label': tf.FixedLenFeature([], tf.int64),
               'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'depth': tf.FixedLenFeature([], tf.int64)}

    datalist = {'image_raw': [], 'label': [], 'height': [], 'width': [], 'depth': []}
    for ser_example in tf.python_io.tf_record_iterator(filename):          # iterate the records from tfr file
        example = tf.parse_single_example(ser_example, features=feature)    # deserialize the serialized example(record)
        img = tf.decode_raw(example['image_raw'], tf.uint8)                 # decode the bytes of string as vector of no
        datalist['image_raw'].append(img)
        datalist['label'].append(example['label'])
        datalist['height'].append(example['height'])
        datalist['width'].append(example['width'])
        datalist['depth'].append(example['depth'])

    sess = tf.Session()                                                     # open a tf session
    sess.run(tf.global_variables_initializer())                             # initialize session global variables
    recv = sess.run(datalist)                                               # get the numpy array by running tf node
    sess.close()                                                            # close tf session

    result = {'image': [], 'label': []}
    for arr in enumerate(recv['image_raw']):
        i = arr[0]
        data = np.array(arr[1]).reshape(recv['height'][i], recv['width'][i], recv['depth'][i])
        result['image'].append(data)
        result['label'].append((recv['label'][i]))

    return result


file_name = "E:\\cifar10\\train\\batch_0.tfrecords"
img_data = tfr_recover_data(file_name)
print(img_data)
print(img_data['image'].__len__(), img_data['label'].__len__())
