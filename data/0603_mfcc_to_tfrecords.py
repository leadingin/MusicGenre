# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0603_mfcc_to_tfrecords.py
@time: 2017/1/10 13:52
"""


# https://github.com/tobegit3hub/deep_recommend_system/blob/master/data/convert_cancer_to_tfrecords.py
import tensorflow as tf
import os
import numpy as np
import math

# The data in cancer.csv:
# 10,10,10,8,6,1,8,9,1,1
# 6,2,1,1,1,1,7,1,1,0
# 2,5,3,3,6,7,7,5,1,1

def get_label(path):
    switcher = {
        'blues' : 0,
        'classical' : 1,
        'country' : 2,
        'disco' : 3,
        'hiphop' : 4,
        'jazz' : 5,
        'metal' : 6,
        'pop' : 7,
        'reggae' : 8,
        'rock' : 9,
    }
    for key in switcher.keys():
        if path.find(key) > -1:
            return switcher.get(key)
    raise NameError("Path error, can't find respective genre.")


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=float)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def convert_mel_to_tfrecords(wavelets_and_spilit_file_folder, spilit_file_path, output_file_path):
    writer = tf.python_io.TFRecordWriter(output_file_path)
    cur_raw_path = ""

    # 将文件内容读入array
    splited_file_array = []
    for line in open(spilit_file_path, 'r'):
        splited_file_array.append(line.strip())

    for root, subdirs, files in os.walk(wavelets_and_spilit_file_folder):
        files.sort()
        # ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # 分别对应[0-9]
        for file in files:
            # 判断文件是否在对应的training set 或者 test set 里面，若是，进行转换
            if file in splited_file_array:

                path = root + "/" + file
                cur_raw_path = path
                if cur_raw_path != "":
                    feature_waves = np.loadtxt(cur_raw_path, dtype=np.float32, delimiter=" ")


                    # onehot encoded label
                    label = get_label(cur_raw_path)
                    label_onehot_encoded = dense_to_one_hot(np.array([label]))

                    # Write each example one by one
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label_onehot_encoded[0])),
                        # (30, 683)
                        "features_raw": tf.train.Feature(float_list=tf.train.FloatList(value=feature_waves)),
                    }))

                    writer.write(example.SerializeToString())
                    print("Successfully converted {} to {}".format(cur_raw_path, output_file_path))

    writer.close()
    print("###############   Finished.   #############")

# convert_tfrecords("scat_data_test.txt", "scat_data_test.tfrecords", "/merge/")

if __name__ == "__main__":

    raw_and_spilit_file_path = 'C:/Users/song/data/mfcc'
    training_set_path = raw_and_spilit_file_path + '/training_set_mfcc.txt'
    test_set_path = raw_and_spilit_file_path+'/test_set_mfcc.txt'

    convert_mel_to_tfrecords(raw_and_spilit_file_path, training_set_path, "merge/mfcc_data_training.tfrecords")
    convert_mel_to_tfrecords(raw_and_spilit_file_path, test_set_path, "merge/mfcc_data_test.tfrecords")