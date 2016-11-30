# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0503_tf_input_batch.py
@time: 11/30/16 4:59 PM
"""
from __future__ import print_function
import tensorflow as tf
import time


def input_pipeline(filenames, batch_size, num_epochs=None):
    features, labels = read_csv_file(filenames, field_delim=",")
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 5 * batch_size
    feature_batch, label_batch = tf.train.shuffle_batch(
        [features, labels], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
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

feature_batch, label_batch = input_pipeline("data/file0.csv", 2) #data/merge/scat_data.txt
with tf.Session() as sess:
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Retrieve a single instance:
    try:
        for i in range(10):
            example, label = sess.run([feature_batch, label_batch])
            print (example)
    except tf.errors.OutOfRangeError:
        print ('Done reading')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()