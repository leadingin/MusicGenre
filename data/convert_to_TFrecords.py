# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: convert_to_TFrecords.py
@time: 12/1/16 5:07 PM
"""

# https://github.com/tobegit3hub/deep_recommend_system/blob/master/data/convert_cancer_to_tfrecords.py
import tensorflow as tf
import os


# The data in cancer.csv:
# 10,10,10,8,6,1,8,9,1,1
# 6,2,1,1,1,1,7,1,1,0
# 2,5,3,3,6,7,7,5,1,1


def convert_tfrecords(input_filename, output_filename):
    current_path = os.getcwd() + "/merge/"
    input_file = os.path.join(current_path, input_filename)
    output_file = os.path.join(current_path, output_filename)
    print("Start to convert {} to {}".format(input_file, output_file))

    writer = tf.python_io.TFRecordWriter(output_file)

    for line in open(input_file, "r"):
        # Split content in CSV file
        data = line.split(" ")
        label = int(data[-1])
        features = [float(i) for i in data[0:-1]]

        # Write each example one by one
        example = tf.train.Example(features=tf.train.Features(feature={
            "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            "features":tf.train.Feature(float_list=tf.train.FloatList(value=features)),
        }))

        writer.write(example.SerializeToString())

    writer.close()
    print("Successfully convert {} to {}".format(input_file, output_file))


convert_tfrecords("scat_data.txt", "scat_data.tfrecords")
