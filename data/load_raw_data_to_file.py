# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: load_raw_data_to_file.py
@time: 2016/11/25 12:44
"""

import os

import sunau
import numpy as np


class LoadRawData:

    def __init__(self):
        pass

    def save_raw_data_to_file(self, raw_data_file_path):
        for root, subdirs, files in os.walk(raw_data_file_path):
            subdirs.sort()
            # ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
            # 分别对应[0-9]
            for subdir in subdirs:
                path = root + "/" + subdir
                for root1, subdirs1, files1 in os.walk(path):
                    for file1 in files1:
                        cur_path = root1 + "/" + file1
                        music = sunau.open(cur_path, 'r')
                        nframes = music.getnframes()
                        # 读取波形数据
                        str_data = music.readframes(nframes)
                        music.close()
                        # 将波形数据转换为数组
                        wave_data = np.fromstring(str_data, dtype=np.short)
                        if wave_data.size != 661794:
                            if wave_data.size > 661794:
                                wave_data = wave_data[:661794]
                            elif wave_data.size < 661794:
                                diff = 661794 - wave_data.size
                                wave_data = np.concatenate((wave_data, np.zeros(diff)))
                        # try:
                        #     f = open("data/rawData.txt", "a")
                        # except:
                        #     exit(1)
                        # str = np.array2string(wave_data, max_line_width = 700000 , formatter={'float_kind':lambda x: "%.4f" % x})
                        # str = str.replace("[", "").replace("]", "")
                        # str += "\n"
                        # try:
                        #     f.write(str)
                        # finally:
                        #     f.close()
                        d = np.array(wave_data)
                        np.savetxt('data/raw/' + subdir + "." + file1 + ".txt", d, fmt='%s', delimiter=' ')
                        print cur_path + " done."

    def convert_single_files(self, origin_file_path, target_file_path):
        of = open(origin_file_path, "r")
        of_str = of.read()
        tf = open(target_file_path, "w")
        tf.write(of_str.replace("\n", " "))
        of.close()
        tf.close()
        print target_file_path + " converted."

    def merge_cvt_files(self, origin_cvt_path, target_file_path):
        of = open(origin_cvt_path, "r")
        of_str = of.read()
        tf = open(target_file_path, "a")
        tf.write(of_str + "\n")
        of.close()
        tf.close()
        print origin_cvt_path + " merged."



'''
test convert_single_files

LoadRawData().convert_single_files("raw/blues.blues.00000.au.txt", "raw/blues.blues.00000.au-cvt.txt")
data = np.loadtxt('raw/blues.blues.00000.au-cvt.txt')
print data.shape
'''

'''
test merge_cvt_files

LoadRawData().merge_cvt_files("raw/blues.blues.00000.au-cvt.txt", "raw/merged_file.txt")
data = np.loadtxt('raw/merged_file.txt')
print data.shape

'''



