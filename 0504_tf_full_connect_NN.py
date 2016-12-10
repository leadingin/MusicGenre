# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0504_tf_full_connect_NN.py
@time: 11/30/16 3:44 PM
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

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
display_step = 1
num_threads = 4
csv_file_path = "data/tvtsets/test_scat_data.txt"
training_file_path = "data/tvtsets/test_scat_data.tfrecords"
batch_size = 20
column_num = count_column_num(csv_file_path, " ")
# file_length = file_len(csv_file_path)
# Network Parameters
n_hidden_1 = 1024  # 1st layer number of features
n_hidden_2 = 1024  # 2nd layer number of features
n_input = column_num - 1
n_classes = 10  # total classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, [batch_size, n_input])
y = tf.placeholder(tf.int32, [batch_size,])



# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer with RELU activation
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

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Launch the graph

audio, label = read_and_decode(training_file_path)

#使用shuffle_batch可以随机打乱输入
audio_batch, label_batch = tf.train.shuffle_batch([audio, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

for i in range(10000):
    # pass it in through the feed_dict
    audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])

    _, loss_val = sess.run([optimizer, cost], feed_dict={x:audio_batch_vals, y:label_batch_vals})
    print (loss_val)



'''
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



    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
'''