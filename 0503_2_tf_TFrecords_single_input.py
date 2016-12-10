# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0503_1_tf_TFrecords_input.py
@time: 12/1/16 7:33 PM
"""

from __future__ import print_function
import tensorflow as tf

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up there dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([], tf.int64),
            'feature': tf.VarLenFeature(tf.float32)
        })
    # now return the converted data
    label = features['label']
    audio = features['feature']
    return label, audio

# returns symbolic label and audio
label, audio = read_and_decode_single_example("data/tvtsets/test_scat_data.tfrecords")

sess = tf.Session()

# Required. See below for explanation
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

# grab examples back.
# first example from file
label_val_1, audio_val_1 = sess.run([label, audio])
# second example from file
label_val_2, audio_val_2 = sess.run([label, audio])

'''
The fact that this works requires a fair bit of effort behind the scenes.
First, it is important to remember that TensorFlow’s graphs contain state.
It is this state that allows the TFRecordReader to remember the location of the tfrecord
it’s reading and always return the next one. This is why for almost all TensorFlow work
we need to initialize the graph. We can use the helper function tf.initialize_all_variables(),
which constructs an op that initializes the state on the graph when you run it.

'''