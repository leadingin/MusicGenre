# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0601_wavelet_CNN.py
@time: 2016/12/24 12:26
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'features_cA': tf.FixedLenFeature([feature_length], tf.float32),
                                           'features_cD': tf.FixedLenFeature([feature_length], tf.float32),
                                       })

    x_cA = tf.cast(features['features_cA'], tf.float32)
    x_cD = tf.cast(features['features_cD'], tf.float32)
    x = tf.concat(0, [x_cA, x_cD])
    y = tf.cast(features['label'], tf.int32)

    return x, y

# Parameters
feature_length = 32768
x_len = feature_length * 2
learning_rate = 0.001
training_epochs = 10000
display_step = 50
num_threads = 8
dropout = 0.75
batch_size = 100

n_classes = 10  # total classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, (batch_size, x_len), name='input_layer')
y = tf.placeholder(tf.int32, (batch_size))
keep_prob = tf.placeholder(tf.float32)  #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, name, strides=1):
    # Conv2D wrapper, with bias and relu activation
    # NHWC
    x = tf.nn.conv2d(x, W, strides=[1, 1, strides, 1], padding="VALID", name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=4):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                          padding="VALID")

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 1, 65536, 1])   # shape = (batch_size, 1, 65536, 1)

    # Convolution Layer
    conv1 = conv2d(x ,weights['wc1'], biases['bc1'], 'conv1', strides=8)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 'conv2', strides=4)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 8, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([1, 4, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([512*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer & correct_prediction
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph

#audio, label = read_and_decode(training_file_path)
#audio_test, label_test = read_and_decode(test_file_path)
features, label = read_and_decode("data/merge/wavelet_data.tfrecords")

#使用shuffle_batch可以随机打乱输入
audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for epoch in range(1000):
        # pass it in through the feed_dict
        audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])

        _, loss_val = sess.run([optimizer, cost], feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: dropout})
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(loss_val))

    print("Training finished.")
    coord.request_stop()
    coord.join(threads)

    # Test model

