# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0504_tf_full_connect_NN.py
@time: 11/30/16 3:44 PM
"""

from __future__ import print_function
import tensorflow as tf
import time

from numpy.distutils.system_info import numarray_info


def input_pipeline(filenames, batch_size, num_threads, num_epochs=None):
    features, labels = read_csv_file(filenames, field_delim=" ")
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 50000
    capacity = min_after_dequeue + 3 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, labels], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue, num_threads=num_threads)
    return feature_batch, label_batch

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


def read_csv_file(filename_queue, field_delim):
    # setup text reader
    column_num = count_column_num(filename_queue, field_delim)
    filename_queue = tf.train.string_input_producer([filename_queue], shuffle=True)
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)

    # setup CSV decoding
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [["null"] for x in range(column_num)]
    t1 = time.time()
    cols = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=field_delim)
    t2 = time.time()
    print("decode csv cost " + str(t2 - t1) + " s.")

    # turn features back into a tensor
    features = tf.pack(cols[0:-1])
    labels = cols[-1]
    return features, labels


# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 50
display_step = 1
num_threads = 4
file_path = "data/merge/scat_data.txt"
column_num = count_column_num(file_path, " ")
file_length = file_len(file_path)
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

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(file_length / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_tensor_x, batch_tensor_y = input_pipeline(file_path, batch_size=batch_size, num_threads=num_threads)
            # Run optimization op (backprop) and cost op (to get loss value)
            batch_x, batch_y = sess.run([batch_tensor_x, batch_tensor_y])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
'''
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
'''