# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0503_1_tf_TFrecords_input.py
@time: 12/1/16 7:33 PM
"""

from __future__ import print_function
import tensorflow as tf

# example: https://github.com/ycszen/tf_lab/blob/master/reading_data/example_tfrecords.py

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           # We know the length of both fields. If not the
                                           # tf.VarLenFeature could be used
                                           'features': tf.FixedLenFeature([8660], tf.float32),
                                       })

    X = tf.cast(features['features'], tf.float32)
    y = tf.cast(features['label'], tf.int32)

    return X, y

img, label = read_and_decode("data/merge/scat_data_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=2, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(10):
        val, l= sess.run([img_batch, label_batch])
        print(val[-10:], l)

