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
                                           'features_cA': tf.FixedLenFeature([32768], tf.float32),
                                           'features_cD': tf.FixedLenFeature([32768], tf.float32),
                                       })

    x_cA = tf.cast(features['features_cA'], tf.float32)
    x_cD = tf.cast(features['features_cD'], tf.float32)
    x = tf.concat(0, [x_cA, x_cD])
    x = tf.reshape(x, (-1, 2))
    y = tf.cast(features['label'], tf.int32)

    return x, y

x, label = read_and_decode("merge/wavelet_data.tfrecords")

#使用shuffle_batch可以随机打乱输入
#测试x_cA
x_batch, label_batch = tf.train.shuffle_batch([x, label],
                                                batch_size=50, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()

# simple model
w = tf.get_variable("w1", [32768*2, 50])
y_pred = tf.matmul(tf.reshape(x_batch,[50, 32768*2]), w)
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
  loss = sess.run([loss_mean])
  print (loss)


'''
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        for i in range(100):
            val, l= sess.run([x_cA_batch, label_batch])
            print(val[-10:], l)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

'''