# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1102_0_mtt_mini_batch_input.py
@time: 2017/1/15 10:33
"""

from __future__ import print_function
import os

MTT_mel_path = 'D:/music_data/mtt_mel'

training_num = 0
validation_num = 0
test_num = 0

none_top50_file_path = 'none_top_50_tags_file.txt'
nt5_file = open(none_top50_file_path, 'r')

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
                    test_num += 1
                elif output_path_subfolder in ['c/']:
                    validation_num += 1
                else:
                    training_num += 1

f = open(none_top50_file_path, 'r')
for line in f:
    cur_folder = line[0]
    if cur_folder in ['d', 'e', 'f']:
        test_num -= 1
    elif output_path_subfolder in ['c']:
        validation_num -= 1
    else:
        training_num -= 1

print(training_num, validation_num, test_num)


'''
result:
all: 18706 1825 5329
top-50: 14951 1825 4332
'''



