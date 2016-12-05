# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0504_tf_full_connect_NN.py
@time: 11/30/16 3:44 PM
"""

from __future__ import print_function
import tensorflow as tf
import time
import numpy as np

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


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
# Parameters
learning_rate = 0.001
training_epochs = 10000
display_step = 1
num_threads = 4
csv_file_path = "data/merge/scat_data_test.txt"
training_file_path = "data/merge/scat_data.tfrecords"
column_num = count_column_num(csv_file_path, " ")
# file_length = file_len(csv_file_path)
# Network Parameters
n_hidden_1 = 256  # 1st layer number of features
n_hidden_2 = 256  # 2nd layer number of features
n_input = column_num - 1
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    t1=time.time()
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all data
        for serialized_example in tf.python_io.tf_record_iterator(training_file_path):
            # Get serialized example from file
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            # Read data in specified format
            label = example.features.feature["label"].int64_list.value
            features = example.features.feature["features"].float_list.value
            # solve error: ValueError: Argument must be a dense tensor, use nparray as input
            features_array = np.array([features])
            features_array = np.reshape(features_array, (1, n_input))
            label_array = dense_to_one_hot(np.array([label]), num_classes = n_classes)
            _, c = sess.run([optimizer, cost], feed_dict={x: features_array, y: label_array})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
    print("Optimization Finished!")
    t2 = time.time()
    print("Training cost: " + str(t2-t1) + " s")

'''
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
'''