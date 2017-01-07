# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0302_0_spilit_wavelet_files.py
@time: 2016/12/26 22:30
"""


import os
import numpy as np

# 每一类音乐的原个数，于之前分割时的命名习惯有关
each_class_source_num = 100

class_name_array = [
        'blues',
        'classical',
        'country',
        'disco',
        'hiphop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock']


def split_training_test(wavelet_path, out_put_path, training_scale):
    training_set_path = out_put_path + '/training_set_raw.txt'
    test_set_path = out_put_path+'/test_set_raw.txt'

    # 删除之前的training_set test_set文件
    if os.path.exists(training_set_path):
        os.remove(training_set_path)
    if os.path.exists(test_set_path):
        os.remove(test_set_path)

    # 创建新的training_set test_set文件并打开
    f_training = open(training_set_path, 'a')
    f_test = open(test_set_path, 'a')

    # 类别的名字，每个类别按照比例生成随机数列，将最终的文件名写入文件
    for class_name in class_name_array:
        index = np.random.permutation(each_class_source_num)
        training_index = index[:int(training_scale * index.size)]
        test_index = index[int(training_scale * index.size):]

        cur_class_training_file_pre_names = []
        for cur_index in training_index:
            if cur_index < 10:
                cur_class_training_file_pre_names.append(class_name + '.0000' + str(cur_index))
            else:
                cur_class_training_file_pre_names.append(class_name + '.000' + str(cur_index))

        cur_class_test_file_pre_names = []
        for cur_index in test_index:
            if cur_index < 10:
                cur_class_test_file_pre_names.append(class_name + '.0000' + str(cur_index))
            else:
                cur_class_test_file_pre_names.append(class_name + '.000' + str(cur_index))
        # test file name prefix
        # for name in cur_class_test_file_pre_names:
        #    print(name)
        for _root, _dirs, files in os.walk(wavelet_path):
            for file in files:
                for training_prefix in cur_class_training_file_pre_names:
                    if file.find(training_prefix) != -1:
                        f_training.write(file + '\n')
                for test_prefix in cur_class_test_file_pre_names:
                    if file.find(test_prefix) != -1:
                        f_test.write(file + '\n')

    f_training.close()
    f_test.close()

    print("Generating index_file of training & test set finished.")


if __name__ == '__main__':
    split_training_test('C:/Users/song/data/raw_spilited', 'C:/Users/song/data/raw_spilited', 0.8)
