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

    def load(self, path, print_lines):
        f = open(path, "r")
        result = [[]]
        line_num = 0
        while True:
            line = f.readline()
            if line:
                line = line.strip()
                arr_str = line.split(" ")
                arr = [np.float32(x) for x in arr_str]
                result.append(arr)
                if print_lines:
                    line_num += 1
                    if line_num % 50 == 0:
                        print "line %i finished" % (line_num)
            else:
                break
        f.close()
        del result[0]
        print "data shape: %i,%i" %(len(result), len(result[0]))
        return result