




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

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    recv = sess.run(datalist)
    sess.close()

    result = {'image': [], 'label': []}
    for arr in enumerate(recv['image_raw']):
        i = arr[0]
        data = np.array(arr[1]).reshape(recv['height'][i], recv['width'][i], recv['depth'][i])
        result['image'].append(data)
        result['label'].append((recv['label'][i]))

    return result



file_name = "E:\\cifar10\\train\\batch_0.tfrecords"
ddd = tfr_recover_data(file_name)
print(ddd)
print(ddd['image'].__len__(),ddd['label'].__len__())

"""with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=1)"""

"""reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })
image = tf.decode_raw(features['image_raw'], tf.uint8)
image.set_shape([3072])
image = tf.cast(image, tf.float32)
v = tf.constant(image)
print(v)"""

'''features1={
          'image_raw': tf.FixedLenFeature([], tf.string)}
features2={
          'label': tf.FixedLenFeature([], tf.int64)}
data1 = []
data2 = []
for s_example in tf.python_io.tf_record_iterator(filename):
 example1 = tf.parse_single_example(s_example, features=features1)
 example2 = tf.parse_single_example(s_example, features=features2)
 img = tf.decode_raw(example1['image_raw'],tf.uint8)
 data1.append(img)

print (data1.__len__())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Y = sess.run([data1])
    print(npY)
    #print(np.array(Y[0]).reshape(32,32,3))
    print(Y[0][0].__len__())'''



#sess = tf.Session()
#print(sess.run(v))





