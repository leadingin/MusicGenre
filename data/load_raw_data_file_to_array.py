# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: load_raw_data_file_to_array.py
@time: 2016/11/27 15:54
"""

from __future__ import print_function
import numpy as np
import re


class LoadRawDataFileToArray:

    def __init__(self):
        pass

    def load(self, path):
        f = open(path, "r")
        counter = 0
        result=[[]]
        for line in f:
            r = []
            counter += 1
            if counter % 50 == 0:
                print ("line %i finished." % (counter))
            for v in line.strip().split(' '):
                try:
                    r.append(np.float32(v))
                # 处理类似\x00这种打印不出来的情况
                except:
                    temp = re.findall(r'[0-9\.\-]', v)
                    v = ''.join(temp)
                    r.append(np.float32(v))
            result.append(r)
        f.close()
        del result[0]
        print ("data shape: %i,%i" %(len(result), len(result[0])))
        return result