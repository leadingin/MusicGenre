# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1200_mtt_2dCNN.py
@time: 2017/2/7 14:55
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
top_50_tags_index = np.loadtxt('data/top_50_tags.txt', delimiter=',', skiprows=0, dtype=int)

# Parameters
x_height = 96
x_width = 1366

# 总共的tag数
n_total_tags = 50
learning_rate = 0.001
training_epochs = 1500 * 150 # 1500 iterations, 100 epochs
display_step = 100
num_threads = 8
dropout = 0.75
#L2_norm = 1e-9
batch_size = 12


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'features_mel': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([n_total_tags], tf.float32),
                                       })

    x = tf.decode_raw(features['features_mel'], tf.float32)
    x = tf.reshape(x, [x_height, x_width, 1])
    y = tf.cast(features['label'], tf.float32)
    return x, y


def load_and_shuffle_to_batch_data(path, batch_size=batch_size):
    features, label = read_and_decode(path)
    # 使用shuffle_batch可以随机打乱输入
    audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                      batch_size=batch_size, capacity=2000,
                                                      min_after_dequeue=1000)
    return audio_batch, label_batch


# tf Graph input
x = tf.placeholder(tf.float32, (batch_size, x_height, x_width, 1), name='input_layer')
y = tf.placeholder(tf.float32, (batch_size, n_total_tags), name='output_layer')
keep_prob = tf.placeholder(tf.float32)  #dropout (keep probability)
# phase_train = tf.placeholder(tf.bool, name='phase_train')


def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


# Create model
def conv_net(x, weights, phase_train=np.array(True)):

    conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
    conv2_1 = tf.nn.relu(batch_norm(conv2_1, 128, phase_train))
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
    conv2_2 = tf.nn.relu(batch_norm(conv2_2, 256, phase_train))
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
    conv2_3 = tf.nn.relu(batch_norm(conv2_3, 512, phase_train))
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
    conv2_4 = tf.nn.relu(batch_norm(conv2_4, 1024, phase_train))
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5)

    conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
    conv2_5 = tf.nn.relu(batch_norm(conv2_5, 2048, phase_train))
    mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_5 = tf.nn.dropout(mpool_5, 0.5)

    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    fc_out = tf.nn.sigmoid(tf.add(tf.matmul(flat, weights['woutput']), weights['boutput']))

    return fc_out

# Store layers weight & bias
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_biases(shape):
    return tf.Variable(tf.zeros(shape))


def get_roc_auc_scores(tags, logits):
    final_acc = 0.
    num = batch_size
    for i in range(batch_size):
        cur_tag_array = tags[i]
        cur_logits_array = logits[i]
        if is_zeros(cur_tag_array):
            if num == 1:
                continue
            else:
                num = num-1
                continue
        roc_auc = roc_auc_score(cur_tag_array, cur_logits_array)
        final_acc += roc_auc
    return final_acc/num


def is_zeros(arr):
    for element in arr:
        if element != 0:
            return False
    return True

weights = {
    'wconv1': init_weights([3, 3, 1, 128]),
    'wconv2': init_weights([3, 3, 128, 256]),
    'wconv3': init_weights([3, 3, 256, 512]),
    'wconv4': init_weights([3, 3, 512, 1024]),
    'wconv5': init_weights([3, 3, 1024, 2048]),
    'bconv1': init_biases([128]),
    'bconv2': init_biases([256]),
    'bconv3': init_biases([512]),
    'bconv4': init_biases([1024]),
    'bconv5': init_biases([2048]),
    'woutput': init_weights([2048, 50]),
    'boutput': init_biases([50])
}


# Construct model
logits = conv_net(x, weights)


# Define loss and optimizer & correct_prediction

# NaN bug
#cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

# cross_entropy_loss with L2 norm
# cross_entropy_loss = -tf.reduce_sum(y * tf.log(logits) + L2_norm * tf.nn.l2_loss(weights['wd1']))
cross_entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y))
tf.scalar_summary("cross_entropy", cross_entropy_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
# load data
audio_batch_training, label_batch_training = load_and_shuffle_to_batch_data("data/merge/mtt_mel_training_filtered.tfrecords", batch_size)
audio_batch_validation, label_batch_validation = load_and_shuffle_to_batch_data("data/merge/mtt_mel_validation_filtered.tfrecords", batch_size)
audio_batch_test, label_batch_test = load_and_shuffle_to_batch_data("data/merge/mtt_mel_test_filtered.tfrecords", batch_size)

# Launch the graph
# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # for TensorBoard
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('model/', sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for epoch in range(int(8000/batch_size)):
    valdation_accuracy_final = 0.
    for epoch in range(training_epochs):
        # pass it in through the feed_dict
        audio_batch_vals_training, label_batch_vals_training = sess.run([audio_batch_training, label_batch_training])
        _, loss_val, pred_ = sess.run([optimizer, cross_entropy_loss, logits], feed_dict={x:audio_batch_vals_training, y:label_batch_vals_training, keep_prob: dropout})

        #print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(loss_val))
        #print(pred_, label_batch_vals_training)

        # calculate accuracy at each step
        if (epoch+1) % display_step == 0:
            audio_batch_validation_vals, label_batch_validation_vals = sess.run([audio_batch_validation, label_batch_validation])
            logits_validation, loss_val_validation = sess.run([logits, cross_entropy_loss], feed_dict={x:audio_batch_validation_vals, y:label_batch_validation_vals, keep_prob:1.0})
            validation_accuracy = get_roc_auc_scores(label_batch_validation_vals, logits_validation)
            print ("training epoch: %d, mini-batch loss: %f, validation accuracy: %f" % ((epoch+1), loss_val, validation_accuracy))
            # print(pred_, label_batch_vals)
            #print(sess.run(weights))

            # add value for Tensorboard at each step
            #summary_str = sess.run(summary_op, feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: 1.0})
            #summary_writer.add_summary(summary_str, (epoch+1))
    #save_path = saver.save(sess, "model/model_2dCNN.ckpt")
    print("#########      Training finished.      #########")

    # Test model
    # batch_test --> reduce_mean --> final_test_accuracy

    test_epochs = 400
    test_accuracy_final = 0.
    for _ in range(test_epochs):
        audio_test_vals, label_test_vals = sess.run([audio_batch_test, label_batch_test])
        logits_test, test_loss_val= sess.run([logits, cross_entropy_loss], feed_dict={x: audio_test_vals, y:label_test_vals, keep_prob: 1.0})
        test_accuracy = get_roc_auc_scores(label_test_vals, logits_test)
        test_accuracy_final += test_accuracy
        print("test epoch: %d, test loss: %f, test accuracy: %f" % (_, test_loss_val, test_accuracy))
    test_accuracy_final /= test_epochs
    print("final test accuracy: %f" % test_accuracy_final)

    coord.request_stop()
    coord.join(threads)
    sess.close()

    # console results is in FCN-5.pdf