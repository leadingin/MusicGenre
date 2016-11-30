# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0502_tf_input_merge_file.py
@time: 11/28/16 5:04 PM
"""

from __future__ import print_function
import tensorflow as tf
import time


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def column_num(fname):
    with open(fname) as f:
        line = f.readline().split(" ")
        # the last column is the class number -->  -1
        return len(line)


filename = "data/merge/scat_data.txt"

# setup text reader
file_length = file_len(filename)
column_num = column_num(filename)
filename_queue = tf.train.string_input_producer([filename])
reader = tf.TextLineReader()
_, csv_row = reader.read(filename_queue)

# setup CSV decoding
# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [["null"] for x in range (column_num)]
t1 = time.time()
cols = tf.decode_csv(csv_row, record_defaults=record_defaults, field_delim=" ")
t2 = time.time()
print ("decode csv cost "+ str(t2-t1) +" s.")

# turn features back into a tensor
features = tf.pack(cols[0:-1])
labels = cols[-1]

print("loading, " + str(file_length) + " line(s)\n")
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # start populating filename queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(file_length):
        # retrieve a single instance
        feature, label = sess.run([features, labels])
        print(feature, label)

    coord.request_stop()
    coord.join(threads)
    print("\ndone loading")