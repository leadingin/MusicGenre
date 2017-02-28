# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1102_1_mtt_top-50-tags_filter.py
@time: 2017/2/19 19:09
"""


from __future__ import print_function
import tensorflow as tf
import numpy as np

# https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

x_height = 96
x_width = 1366

# 总共的tag数
n_tags = 189

top_50_tags_index = np.loadtxt('data/top_50_tags.txt', delimiter=',', skiprows=0, dtype=int)


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'features_mel': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([n_tags], tf.float32),
                                       })

    x = tf.decode_raw(features['features_mel'], tf.float32)
    x = tf.reshape(x, [x_height, x_width])
    y = tf.cast(features['label'], tf.float32)
    return x, y


def get_top_50_tags(top_50_tags_index, tags_batch_val):
    result=[]
    for row in tags_batch_val:
        result_row=[]
        for index in range(len(row)): # 189
            if index in top_50_tags_index:
                result_row.append(row[index])
        result.append(result_row)
    return np.array(result)


def top_50_tags_filter(audio_batch_vals, label_batch_vals):
    # remove none-top-50-tags data
    filtered_indexes = []
    for i in range(audio_batch_vals.shape[0]):
        label_batch_val = label_batch_vals[i]
        # 判断是否所有元素都相同 （a==b）.all()相同则返回true;  (a==b).any()任一相同则返回true
        # 选出不全为0的index
        if (np.array(label_batch_val) != np.zeros((50, ))).any():
            filtered_indexes.append(i)
    audio_batch_vals_filtered = np.array([[]])
    label_batch_vals_filtered = np.array([[]])
    for index in filtered_indexes:
        if audio_batch_vals_filtered.size == 0:
            audio_batch_vals_filtered = np.array([audio_batch_vals[index]])
            label_batch_vals_filtered = np.array([label_batch_vals[index]])
        else:
            audio_batch_vals_filtered = np.vstack((audio_batch_vals_filtered, np.array([audio_batch_vals[index]])))
            label_batch_vals_filtered = np.vstack((label_batch_vals_filtered, np.array([label_batch_vals[index]])))

    return audio_batch_vals_filtered, label_batch_vals_filtered


mel_features, tags = read_and_decode("data/merge/mtt_mel_training.tfrecords")

# 使用shuffle_batch可以随机打乱输入
mel_features_batch, tags_batch = tf.train.shuffle_batch([mel_features, tags],
                                                        batch_size=20, capacity=2000,
                                                        min_after_dequeue=1000)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
tf.train.start_queue_runners(sess=sess)

for i in range(1):
    # pass it in through the feed_dict

    mel_features_batch_val, tags_batch_val = sess.run([mel_features_batch, tags_batch])
   # print(mel_features_batch_val)
  #  print(tags_batch_val)
    top_50_tags_batch_val= get_top_50_tags(top_50_tags_index, tags_batch_val)
   # print(top_50_tags_batch_val)

    print(top_50_tags_filter(mel_features_batch_val, top_50_tags_batch_val))