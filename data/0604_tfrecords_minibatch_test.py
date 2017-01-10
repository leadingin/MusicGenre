# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0604_tfrecords_minibatch_test.py
@time: 2017/1/10 15:21
"""


from __future__ import print_function
import tensorflow as tf

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

n_classes = 10  # total classes (0-9 digits)
x_height = 30
x_width = 683

def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([n_classes], tf.float32),
                                           'features_mfcc': tf.FixedLenFeature([], tf.string),
                                       })

    x = tf.decode_raw(features['features_mfcc'], tf.float32)
    x = tf.reshape(x, [x_height,x_width])
    y = tf.cast(features['label'], tf.float32)
    return x, y

x, label = read_and_decode("merge/mfcc_data_training.tfrecords")

#使用shuffle_batch可以随机打乱输入
#测试x_cA
x_batch, label_batch = tf.train.shuffle_batch([x, label],
                                                batch_size=2, capacity=2000,
                                                min_after_dequeue=1000)
init = tf.global_variables_initializer()
'''
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
            val, l= sess.run([x_batch, label_batch])
            print(val, l)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

