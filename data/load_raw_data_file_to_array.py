# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: load_raw_data_file_to_array.py
@time: 2016/11/27 15:54
"""

import numpy as np


class LoadRawDataFileToArray:

    def __init__(self):
        pass

    def load(self, path):
        f = open(path, "r")
        result = [[np.float32(v) for v in line.strip().split(' ')] for line in f]
        f.close()
        print "data shape: %i,%i" %(len(result), len(result[0]))
        return result