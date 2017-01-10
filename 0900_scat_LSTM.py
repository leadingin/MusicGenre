# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0900_scat_LSTM.py
@time: 2017/1/10 16:45
"""


from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

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

# Parameters
learning_rate = 0.001
training_iters = 100000
training_size = 8000
test_size = 2000
batch_size = 50
display_step = 10

n_classes = 10  # total classes (0-9 digits)
x_height = 96
x_width = 1366

# Network Parameters
n_input = x_width # data input (shape: 96*1366)
n_steps = x_height # timesteps
n_hidden = 128 # hidden layer num of features

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def LSTM(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = LSTM(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# training & test data
features, label = read_and_decode("data/merge/scat_data.tfrecords")
features_test, label_test = read_and_decode("data/merge/scat_data_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

audio_batch_test, label_batch_test = tf.train.shuffle_batch([features_test, label_test],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # for TensorBoard
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('model/', sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: audio_batch_vals, y: label_batch_vals})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: audio_batch_vals, y: label_batch_vals})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: audio_batch_vals, y: label_batch_vals})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Test model
    # batch_test --> reduce_mean --> final_test_accuracy
    test_epochs = int(test_size / batch_size)
    test_accuracy_final = 0.
    for _ in range(test_epochs):
        audio_test_vals, label_test_vals = sess.run([audio_batch_test, label_batch_test])
        test_accuracy = sess.run(accuracy, feed_dict={x: audio_test_vals, y: label_test_vals})
        test_accuracy_final += test_accuracy
        print("test epoch: %d, test accuracy: %f" % (_, test_accuracy))
    test_accuracy_final /= test_epochs
    print("test accuracy %f" % test_accuracy_final)

    coord.request_stop()
    coord.join(threads)
    sess.close()
