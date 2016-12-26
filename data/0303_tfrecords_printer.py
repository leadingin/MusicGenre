# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0303_tfrecords_printer.py
@time: 2016/12/20 20:05
"""

import tensorflow as tf
import os
import math

# Read TFRecords file
current_path = os.getcwd() + "/merge/"
tfrecords_file_name = "wavelet_data.tfrecords"
# tfrecords_file_name = "cancer_test.csv.tfrecords"
input_file = os.path.join(current_path, tfrecords_file_name)

# Constrain the data to print
max_print_number = 100
print_number = 1

for serialized_example in tf.python_io.tf_record_iterator(input_file):
    # Get serialized example from file
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    # Read data in specified format
    label = example.features.feature["label"].int64_list.value
    features_cA = example.features.feature["features_cA"].float_list.value
    features_cD = example.features.feature["features_cD"].float_list.value
    print("Number: {}, label: {} ,\nfeatures_cA: {},\nfeatures_cD: {}".format(print_number, label,
                                                                         features_cA[-10:],  features_cD[-10:]))

    # Return when reaching max print number
    if print_number > max_print_number:
        exit()
    else:
        print_number += 1