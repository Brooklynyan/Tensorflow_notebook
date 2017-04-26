# ==============================================================================
# Author Yan Feng
# Create Date 2017/4/27
# github.com/Brooklynyan/Tensorflow_notebook
# Programmed for Tensorflow Cifar10 dataset CNN tutorial learning
# Windows Platform
# ==============================================================================

""" Cifar10 dataset input pipeline from tfrecords files """


import tensorflow as tf
import os

# CAUTION: all the 2 func contains queue, tf.train.start_queue_runners(sess=sess, coord=coord) must be called
# to ensure the queue will be activated and populated with values
# filename_list should be a list contains all the string format file paths
# num_epochs represents how much times a single file path will be read etc: num_epochs = 1
# if num_epochs is not None, then if the queue is to be run, should call tf.local_variables_initializer()
# num_epochs could be left None value, it will cause infinite loop
# read_tfr_single_record will return a tuple of (image, label) for a single record


def read_tfr_single_record(filename_list, num_epochs):
    filename_queue = tf.train.string_input_producer(filename_list, num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    tfr_key, serialized_example = reader.read(filename_queue)
    raw_data = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)})

    image = tf.decode_raw(raw_data['image_raw'], tf.uint8)
    image = tf.reshape(image, [32, 32, 3])
    label = tf.cast(raw_data['label'], tf.int8)

    return image, label

# record_tup should be an object returned by func read_tfr_single_record
# can replace tf.train.shuffle_batch by tf.train.batch or tf.train.shuffle_batch_join(different conditions)
# return a tuple for one batch, 4D tensor of images shape:[batch_size,height,width,depth].1D of labels:[batch_size]


def read_tfr_batch(record_tup, batch_size, num_threads, capacity, min_after_dequeue):
    image = record_tup[0]
    label = record_tup[1]
    images, labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=num_threads
        , capacity=capacity, min_after_dequeue=min_after_dequeue)
    return images, labels


# data input pipeline using tfrecords file
def input_data_pipeline(filename_list, num_epochs, batch_size, num_threads, capacity, min_after_dequeue):
    record_data = read_tfr_single_record(filename_list, num_epochs)
    return read_tfr_batch(record_data, batch_size, num_threads, capacity, min_after_dequeue)


# get file paths from root
def get_file_pathlist(rootpath):
    pathlist = []
    for dirname, sub, filename in os.walk(rootpath):
        for name in filename:
            path = os.path.join(dirname, name)
            pathlist.append(path)
    return pathlist


# Scripts below is for testing of input data
"""
root = "E:\\Cifar10\\Train"
path_list = get_file_pathlist(root)
dataset = input_data_pipeline(path_list, 1, 500, 4, 2500, 1000)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# for num_epoch
sess.run(tf.local_variables_initializer())
# threads coordinator
coord = tf.train.Coordinator()
# populate and activate queue
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# label counts list
cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

try:
    while not coord.should_stop():

        batch_data = sess.run(dataset)
        print(batch_data)
        for i in batch_data[1]:
            cnt[i] = cnt[i]+1

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop.
    coord.request_stop()

print(cnt)
"""










