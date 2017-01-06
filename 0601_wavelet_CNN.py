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
                                           'label': tf.FixedLenFeature([n_classes], tf.float32),
                                           'features_cA': tf.FixedLenFeature([feature_length], tf.float32),
                                           'features_cD': tf.FixedLenFeature([feature_length], tf.float32),
                                       })

    x_cA = tf.cast(features['features_cA'], tf.float32)
    x_cD = tf.cast(features['features_cD'], tf.float32)
    x = tf.concat(0, [x_cA, x_cD])
    y = tf.cast(features['label'], tf.float32)
    return x, y

# Parameters
feature_length = 32768
x_len = feature_length * 2
learning_rate = 0.001
training_epochs = 10000
display_step = 10
num_threads = 8
dropout = 0.8
batch_size = 10
max_epoches = 8000

n_classes = 10  # total classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, (batch_size, x_len), name='input_layer')
y = tf.placeholder(tf.float32, (batch_size, n_classes), name='output_layer')
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

    # Convolution Layer 1
    conv1 = conv2d(x ,weights['wc1'], biases['bc1'], 'conv1', strides=8)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # shape = (batch_size, 1, 4096, 32)

    # Convolution Layer 2
    # (4096-4+0)/4 + 1 = 1024
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 'conv2', strides=4)
    # Max Pooling (down-sampling)
    # 1024/2 = 512
    conv2 = maxpool2d(conv2, k=2)
    # shape = (batch_size, 1, 512, 64)

    # Convolution Layer 3
    # (512-2+0)/2 + 1 = 256
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 'conv3', strides=2)
    # Max Pooling (down-sampling)
    # 256/2 = 128
    conv3 = maxpool2d(conv3, k=2)
    # shape = (batch_size, 1, 128, 128)

    # Fully connected layer
    # Reshape conv3 output to fit fully connected layer input
    # 128 * 128 = 16384
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    fc_out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    '''
    fc_out 数值很大
    [[ -9.53830238e+12   2.21524703e+13  -1.28374517e+13   1.20256505e+13
       -1.03705634e+13  -6.78045522e+12  -1.48836293e+13   2.55563158e+13
       -2.52714932e+13   2.46169448e+13]
     [ -5.21611090e+12   3.58361921e+13  -1.22493222e+13  -1.00981172e+13
       -8.21462722e+12   7.61901298e+12   5.97017546e+12   2.70052591e+13
       -1.71633195e+13   2.06592646e+12]

    '''
    # fc_out = tf.divide(fc_out, 10e10)


    # softmax output
    out = tf.nn.softmax(fc_out)
    return out


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.scalar_summary("accuracy", accuracy)
    return accuracy

# Store layers weight & bias
weights = {
    # height*width*depth 1*8*1 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([1, 8, 1, 32])),
    # h*w*d 1*4*32 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([1, 4, 32, 64])),
    # h*w*d 1*2*64 conv, 64 inputs, 128 outputs
    'wc3': tf.Variable(tf.random_normal([1, 2, 64, 128])),
    # fully connected, 128*128 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([128*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bc3': tf.Variable(tf.zeros([128])),
    'bd1': tf.Variable(tf.zeros([1024])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
logits = conv_net(x, weights, biases, keep_prob)


# Define loss and optimizer & correct_prediction
# NaN bug
cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(logits, 1e-7, 1.0)))
tf.scalar_summary("cross_entropy", cross_entropy)

# accuracy
acc = accuracy(logits, y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# Launch the graph

features, label = read_and_decode("data/merge/wavelet_data_training.tfrecords")
features_test, label_test = read_and_decode("data/merge/wavelet_data_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()



# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # for TensorBoard
    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('model/', sess.graph_def)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for epoch in range(int(8000/batch_size)):
    for epoch in range(max_epoches):
        # pass it in through the feed_dict
        audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])

        _, loss_val, pred_ = sess.run([optimizer, cross_entropy, logits], feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: dropout})

        #print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(loss_val))
        #print(pred_, label_batch_vals)

        # calculate accuracy at each step
        train_accuracy = sess.run(acc, feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob:1.0})
        print ("step %d, training accuracy %g" % (epoch, train_accuracy))
        print(pred_, label_batch_vals)

        # add value for Tensorboard at each step
        summary_str = sess.run(summary_op, feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: 1.0})
        summary_writer.add_summary(summary_str, epoch)

    print("Training finished.")
    coord.request_stop()
    coord.join(threads)

    # Test model
    correct_num = 0
    test_example_number = 2000
    for _ in range(test_example_number):
        audio_test_val, label_test_val = sess.run([features_test, label_test])
        audio_test_val_vector = np.array([audio_test_val])
        test_pred = conv_net(audio_test_val_vector, weights, biases, 1)

        # predicted label
        predocted_label = sess.run(tf.arg_max([test_pred][0][0], 0))
        # test_label
        test_label = sess.run(tf.arg_max([label_test_val][0], 0))

        print(predocted_label, test_label)
        if predocted_label == test_label:
            correct_num += 1

    print("%i / %i is correct." % (correct_num, test_example_number))
    print("Accuracy is %f ." % (float(correct_num) / test_example_number))
    sess.close()