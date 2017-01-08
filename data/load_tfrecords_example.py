# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: load_tfrecords_example.py
@time: 2017/1/8 20:08
"""

import tensorflow as tf

X = [[1, 2],
     [3, 4],
     [5, 6]]


def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def dump_tfrecord(data, out_file):
    writer = tf.python_io.TFRecordWriter(out_file)
    for x in data:
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'x': _int_feature(x)
            })
        )
        writer.write(example.SerializeToString())
    writer.close()


def load_tfrecord(file_name):
    features = {'x': tf.FixedLenFeature([2], tf.int64)}
    data = []
    for s_example in tf.python_io.tf_record_iterator(file_name):
        example = tf.parse_single_example(s_example, features=features)
        data.append(tf.expand_dims(example['x'], 0))
    return tf.concat(0, data)


if __name__ == "__main__":
    dump_tfrecord(X, 'test_tfrecord')
    print('dump ok')
    data = load_tfrecord('test_tfrecord')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Y = sess.run([data])
        print(Y)