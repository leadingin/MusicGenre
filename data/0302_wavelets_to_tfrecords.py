# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0302_wavelets_to_tfrecords.py
@time: 2016/12/20 19:18
"""

# https://github.com/tobegit3hub/deep_recommend_system/blob/master/data/convert_cancer_to_tfrecords.py
import tensorflow as tf
import os
import numpy as np

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

def convert_wavelets_tfrecords(input_file_folder, output_file_path):
    writer = tf.python_io.TFRecordWriter(output_file_path)
    cur_cA_path = ""
    cur_cD_path = ""
    for root, subdirs, files in os.walk(input_file_folder):
        files.sort()
        # ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # 分别对应[0-9]
        for file in files:
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

                features_cA = [float(i) for i in cA]
                features_cD = [float(i) for i in cD]
                label = get_label(cur_cA_path)
                # Write each example one by one
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    "features_cA": tf.train.Feature(float_list=tf.train.FloatList(value=features_cA)),
                    "features_cD": tf.train.Feature(float_list=tf.train.FloatList(value=features_cD)),
                }))

                writer.write(example.SerializeToString())
                print("Successfully convert {} to {}".format(cur_cA_path[:-1], output_file_path))

    writer.close()
    print("Finished.")

# convert_tfrecords("scat_data_test.txt", "scat_data_test.tfrecords", "/merge/")

if __name__ == "__main__":

    convert_wavelets_tfrecords("C:/Users/song/data/wavelets", "merge/wavelet_data.tfrecords")

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