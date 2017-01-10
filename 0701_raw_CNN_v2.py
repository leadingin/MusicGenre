# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0700_raw_CNN_overfitting.py
@time: 2017/1/7 16:21
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
                                           'features_raw': tf.FixedLenFeature([x_len], tf.float32),
                                       })

    x = tf.cast(features['features_raw'], tf.float32)
    y = tf.cast(features['label'], tf.float32)
    return x, y

# Parameters
x_len = 131072
learning_rate = 0.001
training_epochs = 2000
display_step = 10
num_threads = 8
dropout = 0.75
L2_norm = 1e-9
batch_size = 100
training_size = 8000
test_size = 2000

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
    x = tf.reshape(x, shape=[-1, 1, x_len, 1])   # shape = (batch_size, 1, 131072, 1)

    # Convolution Layer 1
    conv1 = conv2d(x ,weights['wc1'], biases['bc1'], 'conv1', strides=8)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # shape = (batch_size, 1, 8192, 32)

    # Convolution Layer 2
    # (8192-4+0)/4 + 1 = 2048
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 'conv2', strides=4)
    # Max Pooling (down-sampling)
    # 2048/2 = 1024
    conv2 = maxpool2d(conv2, k=2)
    # shape = (batch_size, 1, 1024, 64)

    # Convolution Layer 3
    # (1024-2+0)/2 + 1 = 512
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 'conv3', strides=2)
    # Max Pooling (down-sampling)
    # 512/2 = 256
    conv3 = maxpool2d(conv3, k=2)
    # shape = (batch_size, 1, 256, 128)

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

    fc_out = tf.divide(fc_out, 10e9)
    # softmax output
    out = tf.nn.softmax(fc_out)
    return out


def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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
    # fully connected, 256*128 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([256*128, 512])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([512, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([32])),
    'bc2': tf.Variable(tf.zeros([64])),
    'bc3': tf.Variable(tf.zeros([128])),
    'bd1': tf.Variable(tf.zeros([512])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
logits = conv_net(x, weights, biases, keep_prob)


# Define loss and optimizer & correct_prediction

# NaN bug
#cross_entropy = -tf.reduce_sum(y * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))

# cross_entropy_loss with L2 norm
# cross_entropy_loss = -tf.reduce_sum(y * tf.log(logits) + L2_norm * tf.nn.l2_loss(weights['wd1']))
cross_entropy_loss = -tf.reduce_sum(y * tf.log(logits))
tf.scalar_summary("cross_entropy", cross_entropy_loss)

# accuracy
acc = accuracy(logits, y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

# Launch the graph

features, label = read_and_decode("data/merge/raw_data_training.tfrecords")
features_test, label_test = read_and_decode("data/merge/raw_data_test.tfrecords")

#使用shuffle_batch可以随机打乱输入
audio_batch, label_batch = tf.train.shuffle_batch([features, label],
                                                batch_size=batch_size, capacity=2000,
                                                min_after_dequeue=1000)

audio_batch_test, label_batch_test = tf.train.shuffle_batch([features_test, label_test],
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
    summary_writer = tf.train.SummaryWriter('model/', sess.graph)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # for epoch in range(int(8000/batch_size)):
    for epoch in range(training_epochs):
        # pass it in through the feed_dict
        audio_batch_vals, label_batch_vals = sess.run([audio_batch, label_batch])

        _, loss_val, pred_ = sess.run([optimizer, cross_entropy_loss, logits], feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: dropout})

        #print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(loss_val))
        #print(pred_, label_batch_vals)

        # calculate accuracy at each step
        if (epoch+1) % display_step == 0:
            train_accuracy = sess.run(acc, feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob:1.0})
            print ("training epoch: %d, mini-batch loss: %f, mini-batch training accuracy: %f" % ((epoch+1), loss_val, train_accuracy))
            # print(pred_, label_batch_vals)
            #print(sess.run(weights))

            # add value for Tensorboard at each step
            #summary_str = sess.run(summary_op, feed_dict={x:audio_batch_vals, y:label_batch_vals, keep_prob: 1.0})
            #summary_writer.add_summary(summary_str, (epoch+1))
    save_path = saver.save(sess, "model/model_cnn_raw_data.ckpt")
    print("#########      Training finished && model saved.      #########")

    # Test model
    # batch_test --> reduce_mean --> final_test_accuracy

    test_epochs = int(test_size / batch_size)
    test_accuracy_final = 0.
    for _ in range(test_epochs):
        audio_test_vals, label_test_vals = sess.run([audio_batch_test, label_batch_test])
        test_accuracy = sess.run(acc, feed_dict={x: audio_test_vals, y: label_test_vals, keep_prob: 1.0})
        test_accuracy_final += test_accuracy
        print("test epoch: %d, test accuracy: %f" % (_, test_accuracy))
    test_accuracy_final /= test_epochs
    print("test accuracy %f" % test_accuracy_final)

    coord.request_stop()
    coord.join(threads)
    sess.close()
