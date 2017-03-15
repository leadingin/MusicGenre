# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0702_MTT_mel_to_tfrecords.py
@time: 2017/1/14 18:52
"""


import numpy as np
import os
import tensorflow as tf


MTT_mel_path = 'D:/music_data/mtt_mel_single'

mtt_tags = np.loadtxt('MTT_tags.csv', delimiter=',', skiprows=1, dtype=str)
#print(mtt_tags.shape)
print('Tag file loaded.')

single_file = 'merge/mtt_mel_single.tfrecords'


def mel_to_tfrecords(MTT_mel_path, tfrecords_file):
    writer = tf.python_io.TFRecordWriter(tfrecords_file)

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
                        pass
                        #right_tfrecords_file = training_file
                    elif output_path_subfolder in ['c/']:
                        pass
                        #right_tfrecords_file = validation_file

                    else:
                        right_tfrecords_file = single_file

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
                                break
    writer.close()
    print("###############  {} Finished.   #############".format(tfrecords_file))




if __name__ == "__main__":
    mel_to_tfrecords(MTT_mel_path, single_file)
