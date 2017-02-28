# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1304_mtt_multi-layer_LSTM_OOM.py
@time: 2017/2/27 21:10

https://medium.com/@erikhallstrm/using-the-dynamicrnn-api-in-tensorflow-7237aba7f7ea#.qduigey12
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score


x_height = 96
x_width = 1366
# 总共的tag数
n_tags = 50

learning_rate = 0.001
training_epochs = 8000 * 150 # 8000 iterations, 150 epochs
display_step = 100
total_series_length = x_height * x_width
state_size = 512
batch_size = 2
num_layers = 2


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
    x = tf.reshape(x, (total_series_length,))
    y = tf.cast(features['label'], tf.float32)
    return x, y


def load_and_shuffle_to_batch_data(path, batch_size=batch_size):
    features, label = read_and_decode(path)
    # 使用shuffle_batch可以随机打乱输入
    audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                      batch_size=batch_size, capacity=2000,
                                                      min_after_dequeue=1000)
    return audio_batch, label_batch


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


# placeholders
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, total_series_length])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, n_tags])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unpack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

W2 = tf.Variable(np.random.rand(state_size, n_tags), dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, n_tags)), dtype=tf.float32)

# Forward passes
cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
states_series, current_state = tf.nn.dynamic_rnn(cell, tf.expand_dims(batchX_placeholder, -1), initial_state=rnn_tuple_state)
# states_series = tf.reshape(states_series, [-1, state_size])

logits = tf.matmul(current_state[-1][0], W2) + b2  #Broadcasted addition  # 取最上一层的c值计算logits
labels = tf.reshape(batchY_placeholder, [-1, n_tags])

# logits_series = tf.unpack(tf.reshape(logits, [batch_size, truncated_backprop_length, n_tags]), axis=1)
# predictions_series = [tf.nn.softmax(logit) for logit in logits_series]


losses = tf.nn.softmax_cross_entropy_with_logits(logits, batchY_placeholder)
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

audio_batch_training, label_batch_training = load_and_shuffle_to_batch_data("data/merge/mtt_mel_training_filtered.tfrecords", batch_size)
audio_batch_validation, label_batch_validation = load_and_shuffle_to_batch_data("data/merge/mtt_mel_validation_filtered.tfrecords", batch_size)
audio_batch_test, label_batch_test = load_and_shuffle_to_batch_data("data/merge/mtt_mel_test_filtered.tfrecords", batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    _current_state = np.zeros((num_layers, 2, batch_size, state_size))
    valdation_accuracy_final = 0.
    for iteration_idx in range(training_epochs):
        audio_batch_vals_training, label_batch_vals_training = sess.run([audio_batch_training, label_batch_training])
        _train_step, _total_loss, _current_state = sess.run([train_step, total_loss, current_state], feed_dict={batchX_placeholder: audio_batch_vals_training,
                                                 batchY_placeholder: label_batch_vals_training,
                                                 init_state: _current_state})

        print("iter %d, loss: %f") % ((iteration_idx + 1), _total_loss)

        if (iteration_idx + 1) % display_step == 0:
            validation_epochs = 100
            cur_validation_acc = 0.
            for _ in range(validation_epochs):
                audio_batch_validation_vals, label_batch_validation_vals = sess.run([audio_batch_validation, label_batch_validation])
                logits_validation, loss_val_validation = sess.run([logits, total_loss],
                                                                  feed_dict={batchX_placeholder: audio_batch_validation_vals,
                                                                             batchY_placeholder: label_batch_validation_vals})
                validation_accuracy = get_roc_auc_scores(label_batch_validation_vals, logits_validation)
                cur_validation_acc += validation_accuracy
                # print("test iter: %d, test loss: %f, test accuracy: %f" % (_, test_loss_val, test_accuracy))
            cur_validation_acc /= validation_epochs
            print("training iter: %d, mini-batch loss: %f, validation accuracy: %f" % (
            (iteration_idx + 1), _total_loss, validation_accuracy))

    print("#########      Training finished && model saved.      #########")

    # Test model
    # batch_test --> reduce_mean --> final_test_accuracy

    test_epochs = 400
    test_accuracy_final = 0.
    for _ in range(test_epochs):
        audio_test_vals, label_test_vals = sess.run([audio_batch_test, label_batch_test])
        logits_test, test_loss_val = sess.run([logits, total_loss],
                                              feed_dict={batchX_placeholder: audio_test_vals,
                                                         batchY_placeholder: label_test_vals})
        test_accuracy = get_roc_auc_scores(label_test_vals, logits_test)
        test_accuracy_final += test_accuracy
        print("test epoch: %d, test loss: %f, test accuracy: %f" % (_, test_loss_val, test_accuracy))
    test_accuracy_final /= test_epochs
    print("final test accuracy: %f" % test_accuracy_final)

    coord.request_stop()
    coord.join(threads)
    sess.close()


