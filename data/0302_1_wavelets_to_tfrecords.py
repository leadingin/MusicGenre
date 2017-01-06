# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0302_1_wavelets_to_tfrecords.py
@time: 2016/12/20 19:18
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


def convert_wavelets_to_tfrecords(wavelets_and_spilit_file_folder, spilit_file_path, output_file_path):
    writer = tf.python_io.TFRecordWriter(output_file_path)
    cur_cA_path = ""
    cur_cD_path = ""

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
                if path.find("cA") > -1:
                    cur_cA_path = path
                elif path.find("cD") > -1:
                    cur_cD_path = path
                if cur_cA_path != "" and cur_cA_path[:-1]==cur_cD_path[:-1]:
                    print("Start to convert {} to {}".format(cur_cA_path[:-1], output_file_path))
                    cA = np.loadtxt(cur_cA_path, dtype=np.float32, delimiter="\n")
                    cD = np.loadtxt(cur_cD_path, dtype=np.float32, delimiter="\n")
                    wavelets = np.vstack((cA, cD))

                    features_cA = []
                    for i in cA:
                        val = float(i)
                        if math.isnan(val):
                            features_cA.append(0.)
                        else:
                            features_cA.append(val)

                    features_cD = []
                    for i in cD:
                        val = float(i)
                        if math.isnan(val):
                            features_cD.append(0.)
                        else:
                            features_cD.append(val)

                    label = get_label(cur_cA_path)
                    label_onehot_encoded = dense_to_one_hot(np.array([label]))
                    # Write each example one by one
                    example = tf.train.Example(features=tf.train.Features(feature={
                        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label_onehot_encoded[0])),
                        "features_cA": tf.train.Feature(float_list=tf.train.FloatList(value=features_cA)),
                        "features_cD": tf.train.Feature(float_list=tf.train.FloatList(value=features_cD)),
                    }))

                    writer.write(example.SerializeToString())
                    print("Successfully converted {} to {}".format(cur_cA_path[:-1], output_file_path))

    writer.close()
    print("###############   Finished.   #############")

# convert_tfrecords("scat_data_test.txt", "scat_data_test.tfrecords", "/merge/")

if __name__ == "__main__":

    wavelets_and_spilit_file_path = 'C:/Users/song/data/wavelets'
    training_set_path = wavelets_and_spilit_file_path + '/training_set.txt'
    test_set_path = wavelets_and_spilit_file_path+'/test_set.txt'

    convert_wavelets_to_tfrecords(wavelets_and_spilit_file_path, training_set_path, "merge/wavelet_data_training.tfrecords")
    convert_wavelets_to_tfrecords(wavelets_and_spilit_file_path, test_set_path, "merge/wavelet_data_test.tfrecords")

    '''
    # test_get_label()

    for root, subdirs, files in os.walk("C:/Users/song/data/wavelets"):
        files.sort()
        # ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # 分别对应[0-9]
        for file in files:
            path = root + "/" + file
            print(get_label(path), path)
    '''