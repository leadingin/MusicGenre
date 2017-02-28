# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1102_1_mtt_top-50-tags_filter.py
@time: 2017/2/19 19:09
"""


import numpy as np
import os
import tensorflow as tf

top_50_tags_index = np.loadtxt('data/top_50_tags.txt', delimiter=',', skiprows=0, dtype=int)
MTT_mel_path = 'D:/music_data/mtt_mel'

mtt_tags = np.loadtxt('data/MTT_tags.csv', delimiter=',', skiprows=1, dtype=str)
#print(mtt_tags.shape)
print('Tag file loaded.')


training_file = 'data/merge/mtt_mel_training_filtered.tfrecords'
validation_file = 'data/merge/mtt_mel_validation_filtered.tfrecords'
test_file = 'data/merge/mtt_mel_test_filtered.tfrecords'

def is_zeros(arr):
    for element in arr:
        if element != 0:
            return False
    return True

def get_top_50_tags(top_50_tags_index, tags):
    result = []
    for index in range(len(tags)):  # 189
        if index in top_50_tags_index:
            result.append(tags[index])
    return result


def mel_to_tfrecords(MTT_mel_path, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)

    training_num = 0
    validation_num = 0
    test_num = 0

    for root, subdirs, files in os.walk(MTT_mel_path):
        subdirs.sort()
        for subdir in subdirs:
            path = root + "/" + subdir
            for root1, subdirs1, files1 in os.walk(path):
                for file1 in files1:
                    cur_path = root1 + "/" + file1
                    output_path_subfolder = cur_path[:(cur_path.rfind("/") + 1)]
                    output_path_subfolder = output_path_subfolder[-2:]

                    if output_path_subfolder in ['d/', 'e/', 'f/']:
                        right_tfrecords_file = test_file
                        test_num += 1
                    elif output_path_subfolder in ['c/']:
                        right_tfrecords_file = validation_file
                        validation_num += 1
                    else:
                        right_tfrecords_file = training_file
                        training_num += 1

                    if tfrecords_file == right_tfrecords_file:
                        feature_waves = np.loadtxt(cur_path, dtype=np.float32, delimiter=" ")
                        feature_bytes = feature_waves.tobytes()

                        # 找当前.mel_spectrogram文件对应的tag
                        last_colum_str = (output_path_subfolder + file1).replace('.mel_spectrogram', '.wav')
                        for row in mtt_tags:
                            last_colum_bytes = row[-1] # type: <numpy.str_>  val: b'xxx'
                            # str(last_colum_bytes) = 'b'f/american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.wav''
                            # last_colum_bytes_to_str='f/american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.wav'
                            last_colum_bytes_to_str = str(last_colum_bytes)[2:-1]
                            # print('f/american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.wav'==last_colum_bytes_to_str)
                            if last_colum_str == last_colum_bytes_to_str:
                                cur_tags = row[:-1]
                                tag = []
                                for bytes in cur_tags:
                                    b_str = str(bytes)[2:-1]
                                    tag.append(float(b_str))

                                tag = get_top_50_tags(top_50_tags_index, tag)
                                if not is_zeros(tag):
                                    # print(len(tag))
                                    # 找到tag后写入feature和label
                                    example = tf.train.Example(features=tf.train.Features(feature={
                                        # len=189
                                        "label": tf.train.Feature(float_list=tf.train.FloatList(value=tag)),
                                        # (96, 1366)
                                        "features_mel": tf.train.Feature(
                                            bytes_list=tf.train.BytesList(value=[feature_bytes])),

                                    }))

                                    writer.write(example.SerializeToString())
                                    print("Successfully converted {} to {}".format(last_colum_str, tfrecords_file))
                                else:
                                    none_top_50_tags_files = open("none_top_50_tags_file.txt", "a")
                                    none_top_50_tags_files.write(last_colum_str +"\n")
                                    none_top_50_tags_files.close()
                                    if output_path_subfolder in ['d/', 'e/', 'f/']:
                                        test_num -= 1
                                    elif output_path_subfolder in ['c/']:
                                        validation_num -= 1
                                    else:
                                        training_num -= 1
                                break
    writer.close()
    print("###############  {} Finished.   #############".format(tfrecords_file))
    print("training_num:", training_num)
    print("validation_num:", validation_num)
    print("test_num:", test_num)




if __name__ == "__main__":
    mel_to_tfrecords(MTT_mel_path, training_file)
    mel_to_tfrecords(MTT_mel_path, validation_file)
    mel_to_tfrecords(MTT_mel_path, test_file)