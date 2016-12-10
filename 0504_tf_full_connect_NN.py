# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0504_tf_full_connect_NN.py
@time: 11/30/16 3:44 PM
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf


def csv_file_line_number(fname):
    with open(fname, "r") as f:
        num = 0
        for line in f:
            num += 1
    return num


def count_column_num(fname, field_delim):
    with open(fname) as f:
        line = f.readline().split(field_delim)
        # the last column is the class number -->  -1
        return len(line)


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
                                           # We know the length of both fields. If not the
                                           # tf.VarLenFeature could be used
                                           'features': tf.FixedLenFeature([n_input], tf.float32),
                                       })

    X = tf.cast(features['features'], tf.float32)
    y = tf.cast(features['label'], tf.int32)

    return X, y


# Parameters
learning_rate = 0.001
training_epochs = 10000
display_step = 50
num_threads = 8

training_csv_file_path = "data/tvtsets/training_scat_data.txt"
training_file_path = "data/tvtsets/training_scat_data.tfrecords"
test_csv_file_path = "data/tvtsets/test_scat_data.txt"
test_file_path = "data/tvtsets/test_scat_data.tfrecords"
batch_size = 20
column_num = count_column_num(training_csv_file_path, " ")
# file_length = file_len(csv_file_path)
# Network Parameters
n_hidden_1 = 1024  # 1st layer number of features
n_hidden_2 = 1024  # 2nd layer number of features

# h1:512 h2:512, acc is about:0.4


n_input = column_num - 1
n_classes = 10  # total classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, [batch_size, n_input])
y = tf.placeholder(tf.int32, [batch_size,])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with sigmoid activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with sigmoid activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer & correct_prediction
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph

audio, label = read_and_decode(training_file_path)
audio_test, label_test = read_and_decode(test_file_path)


#使用shuffle_batch可以随机打乱输入
audio_batch, label_batch = tf.train.shuffle_batch([audio, label],
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
    for epoch in range(10000):
        # pass it in through the feed_dict
        audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])

        _, loss_val = sess.run([optimizer, cost], feed_dict={x:audio_batch_vals, y:label_batch_vals})
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(loss_val))

    print("Training finished.")
    coord.request_stop()
    coord.join(threads)

    # Test model
    # Calculate accuracy
    test_example_number = csv_file_line_number(test_csv_file_path)
    correct_num = 0
    for _ in range(test_example_number):
        audio_test_val, label_test_val = sess.run([audio_test, label_test])
        audio_test_val_vector = np.array([audio_test_val])
        test_pred = multilayer_perceptron(audio_test_val_vector, weights, biases)
        '''
        print (sess.run([test_pred]))

        [array([[ 11.13519478,  12.56501865,  14.68154907,   2.02798128,
         -5.89219952,  -0.76298785,   0.46614531,   5.27717066,
          7.54774714,   7.12729597]], dtype=float32)]
        '''
        pred_class_index = sess.run(tf.argmax(test_pred, 1))
        '''
        print (sess.run(tf.argmax(test_pred, 1)))

        [8]
        0
        [5]
        0
        [9]
        0
        [8]
        0
         ....

        [9]
        9
        [9]
        9
        [4]
        9
        [1]
        9
        '''

        if (label_test_val == pred_class_index[0]):
            correct_num += 1
    print (("%i / %i is correct.") % (correct_num, test_example_number))
    print (("Accuracy is %f .") % (float(correct_num) / test_example_number))
    sess.close()