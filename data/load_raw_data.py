# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: load_raw_data.py
@time: 2016/11/25 12:44
"""

import os

import sunau
import numpy as np


class LoadRawData:

    def __init__(self):
        pass

    def load(self, file_path):
        data = np.array([])

        for root, subdirs, files in os.walk(file_path):
            for subdir in subdirs:
                path = root + "\\" + subdir
                for root1, subdirs1, files1 in os.walk(path):
                    for file1 in files1:
                        music = sunau.open(root1 + "\\" + file1, 'r')
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

                        if data.size == 0:
                            data = wave_data
                        else:
                            data = np.vstack((data, wave_data))

        print data.shape
        return data


