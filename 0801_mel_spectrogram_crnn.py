# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0801_mel_spectrogram_crnn.py
@time: 2017/2/18 15:42
"""

import tensorflow as tf
import sklearn.metrics as sm
import numpy as np

batch_size = 20
learning_rate = 0.003
n_epoch = 400

n_classes = 10  # total classes (0-9 digits)
x_height = 96
x_width = 1366


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_biases(shape):
    return tf.Variable(tf.zeros(shape))


def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
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


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([n_classes], tf.float32),
                                           'features_mel': tf.FixedLenFeature([], tf.string),
                                       })

    x = tf.decode_raw(features['features_mel'], tf.float32)
    x = tf.reshape(x, [x_height,x_width, 1]) # 与placeholder 保持一致
    y = tf.cast(features['label'], tf.float32)
    return x, y



def crnn(melspectrogram, weights, phase_train):
    x = tf.cast(tf.pad(melspectrogram, [[0, 0], [0, 0], [37, 37], [0, 0]], 'CONSTANT'), tf.float32)
    x = batch_norm(tf.reshape(x, [-1, 1, 96, 1440]), 1440, phase_train)
    x = tf.reshape(x, [-1, 96, 1440, 1])
    conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
    conv2_1 = tf.nn.relu(batch_norm(conv2_1, 64, phase_train))
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'),
                     weights['bconv2'])
    conv2_2 = tf.nn.relu(batch_norm(conv2_2, 128, phase_train))
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'),
                     weights['bconv3'])
    conv2_3 = tf.nn.relu(batch_norm(conv2_3, 128, phase_train))
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'),
                     weights['bconv4'])
    conv2_4 = tf.nn.relu(batch_norm(conv2_4, 128, phase_train))
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5)

    gru1_in = tf.reshape(dropout_4, [-1, 15, 128])
    gru1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 15)
    gru1_out, state = tf.nn.dynamic_rnn(gru1, gru1_in, dtype=tf.float32, scope='gru1')

    gru2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 15)
    gru2_out, state = tf.nn.dynamic_rnn(gru2, gru1_out, dtype=tf.float32, scope='gru2')
    gru2_out = tf.transpose(gru2_out, [1, 0, 2])
    gru2_out = tf.gather(gru2_out, int(gru2_out.get_shape()[0]) - 1)
    dropout_5 = tf.nn.dropout(gru2_out, 0.3)

    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat, weights['woutput']), weights['boutput']))
    return p_y_X


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]


if __name__ == '__main__':

    features, label = read_and_decode("data/merge/mel_data_training.tfrecords")
    features_test, label_test = read_and_decode("data/merge/mel_data_test.tfrecords")

    # 使用shuffle_batch可以随机打乱输入
    audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                      batch_size=batch_size, capacity=2000,
                                                      min_after_dequeue=1000)

    audio_batch_test, label_batch_test = tf.train.shuffle_batch([features_test, label_test],
                                                                batch_size=batch_size, capacity=2000,
                                                                min_after_dequeue=1000)


    weights = {
        'wconv1': init_weights([3, 3, 1, 64]),
        'wconv2': init_weights([3, 3, 64, 128]),
        'wconv3': init_weights([3, 3, 128, 128]),
        'wconv4': init_weights([3, 3, 128, 128]),
        'bconv1': init_biases([64]),
        'bconv2': init_biases([128]),
        'bconv3': init_biases([128]),
        'bconv4': init_biases([128]),
        'woutput': init_weights([32, 10]),
        'boutput': init_biases([10])}

    X = tf.placeholder("float", [None, 96, 1366, 1])
    y = tf.placeholder("float", [None, 10])
    lrate = tf.placeholder("float")
    phase_train = np.array(True)

    y_ = crnn(X, weights, phase_train)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    predict_op = y_

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        # Start input enqueue threads.
        # 缺少这2行， audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch]) 程序会假死
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)


        for i in range(n_epoch):
            audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])
            train_input_dict = {X: audio_batch_vals,
                                y: label_batch_vals,
                                lrate: learning_rate
                                #phase_train: True,
                                }
            epoch_logits, epoch_cost, _ = sess.run([y_, cost, train_op], feed_dict=train_input_dict)
            print('Training Epoch : ', i+1, 'cost : ', epoch_cost)
            print(epoch_logits)
            #print sort_result(tags, predictions)[:5])

        # test model
        test_epochs = int(200/batch_size)
        final_roc_auc_score = 0.
        for j in range(test_epochs):

            audio_test_vals, label_test_vals = sess.run([audio_batch_test, label_batch_test])
            test_input_dict = {X: audio_test_vals,
                               y: label_test_vals
                               # phase_train: True,
                               }

            predictions = sess.run(predict_op, feed_dict=test_input_dict)
            cur_epoch_roc_auc_score = sm.roc_auc_score(label_test_vals, predictions, average='samples')
            final_roc_auc_score += cur_epoch_roc_auc_score
            print('Test Epoch : ', j+1, 'AUC : ', cur_epoch_roc_auc_score)

        print('Test Auc is : ', final_roc_auc_score/test_epochs)


