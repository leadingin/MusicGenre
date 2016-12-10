# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0503_1_tf_TFrecords_input.py
@time: 12/1/16 7:33 PM
"""

from __future__ import print_function
import tensorflow as tf

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

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

img, label = read_and_decode("data/tvtsets/test_scat_data.tfrecords")

#使用shuffle_batch可以随机打乱输入
img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                batch_size=20, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()

# simple model
w = tf.get_variable("w1", [8660, 10])
y_pred = tf.matmul(img_batch, w)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, label_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

for i in range(200):
  # pass it in through the feed_dict
  _, loss_val = sess.run([train_op, loss_mean])
  print (loss_val)


'''
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for i in range(10):
            val, l= sess.run([img_batch, label_batch])
            print(val[-10:], l)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
'''
