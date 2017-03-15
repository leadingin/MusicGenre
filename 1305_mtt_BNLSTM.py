# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1305_mtt_BNLSTM.py
@time: 2017/2/28 16:54
"""

# -*- coding:utf-8 -*-
from __future__ import print_function

import tensorflow as tf
from sklearn.metrics import roc_auc_score
import bnlstm

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
from tensorflow.python.ops.rnn import dynamic_rnn

batch_size = 10
num_steps = 96 # number of truncated backprop steps
state_size = 512
learning_rate = 0.001
training_epochs = 1500 * 150 # 1500 iterations, 150 epochs
display_step = 100
dropout = 0.75

x_height = 96
x_width = 1366
# 总共的tag数
n_tags = 50


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'features_mel': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([n_tags], tf.float32),
                                       })

    x = tf.decode_raw(features['features_mel'], tf.float32)
    x = tf.reshape(x, [x_height, x_width])
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
x = tf.placeholder(tf.float32, (batch_size, num_steps, x_width), name='input_placeholder')
y = tf.placeholder(tf.float32, (batch_size, n_tags), name='labels_placeholder')

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([state_size, n_tags]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_tags]))
}
# model
def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, x_width])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, num_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = bnlstm.BNLSTMCell(state_size, True)
    tf.zeros_initializer()
    # c, h
    initialState = (
        tf.random_normal([batch_size, state_size], stddev=0.1),
        tf.random_normal([batch_size, state_size], stddev=0.1))

    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout)
    # Get lstm cell output
    outputs, states = tf.nn.rnn(lstm_cell, x,initial_state=initialState, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

'''
init_state = tf.nn.rnn_cell.LSTMStateTuple(tf.random_normal([batch_size, state_size], stddev=0.1),
              tf.random_normal([batch_size, state_size], stddev=0.1)) # lstm_tuple(c, h)

cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
x = tf.reshape(x, (batch_size, x_height * x_width, 1))
outputs, state = dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)
'''
def is_zeros(arr):
    for element in arr:
        if element != 0:
            return False
    return True


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

# load data
audio_batch_training, label_batch_training = load_and_shuffle_to_batch_data("data/merge/mtt_mel_training_filtered.tfrecords", batch_size)
audio_batch_validation, label_batch_validation = load_and_shuffle_to_batch_data("data/merge/mtt_mel_validation_filtered.tfrecords", batch_size)
audio_batch_test, label_batch_test = load_and_shuffle_to_batch_data("data/merge/mtt_mel_test_filtered.tfrecords", batch_size)

logits = RNN(x, weights, biases)
cross_entropy_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    # for TensorBoard
    #summary_op = tf.merge_all_summaries()
    #summary_writer = tf.train.SummaryWriter('model/', sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # for epoch in range(int(8000/batch_size)):
    valdation_accuracy_final = 0.
    for epoch in range(training_epochs):
        # pass it in through the feed_dict
        audio_batch_vals_training, label_batch_vals_training = sess.run([audio_batch_training, label_batch_training])
        _, loss_val, pred_ = sess.run([optimizer, cross_entropy_loss, logits], feed_dict={x:audio_batch_vals_training, y:label_batch_vals_training})

        #print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(loss_val))
        #print(pred_, label_batch_vals_training)

        if (epoch + 1) % display_step == 0:
            validation_epochs = 100
            cur_validation_acc = 0.
            for _ in range(validation_epochs):
                audio_batch_validation_vals, label_batch_validation_vals = sess.run(
                    [audio_batch_validation, label_batch_validation])
                logits_validation, loss_val_validation = sess.run([logits, cross_entropy_loss],
                                                                  feed_dict={x: audio_batch_validation_vals,
                                                                             y: label_batch_validation_vals})
                validation_accuracy = get_roc_auc_scores(label_batch_validation_vals, logits_validation)
                cur_validation_acc += validation_accuracy
                # print("test iter: %d, test loss: %f, test accuracy: %f" % (_, test_loss_val, test_accuracy))
            cur_validation_acc /= validation_epochs
            print("training iter: %d, mini-batch loss: %f, validation accuracy: %f" % (
            (epoch + 1), loss_val, validation_accuracy))
            # print(pred_, label_batch_vals)
            # print(sess.run(weights))

            # add value for Tensorboard at each step
            # summary_str = sess.run(summary_op, feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: 1.0})
            # summary_writer.add_summary(summary_str, (epoch+1))
            # save_path = saver.save(sess, "model/model_2dCNN.ckpt")
    print("#########      Training finished && model saved.      #########")

    # Test model
    # batch_test --> reduce_mean --> final_test_accuracy

    test_epochs = 400
    test_accuracy_final = 0.
    for _ in range(test_epochs):
        audio_test_vals, label_test_vals = sess.run([audio_batch_test, label_batch_test])
        logits_test, test_loss_val= sess.run([logits, cross_entropy_loss], feed_dict={x: audio_test_vals, y:label_test_vals})
        test_accuracy = get_roc_auc_scores(label_test_vals, logits_test)
        test_accuracy_final += test_accuracy
        print("test epoch: %d, test loss: %f, test accuracy: %f" % (_, test_loss_val, test_accuracy))
    test_accuracy_final /= test_epochs
    print("final test accuracy: %f" % test_accuracy_final)

    coord.request_stop()
    coord.join(threads)
    sess.close()