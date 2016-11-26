# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0103_test_cv_single_file.py
@time: 2016/11/23 20:11
"""
import numpy as np

data = np.loadtxt('data/testFeatureDataSingleFile.txt')
print data
print data.shape

data1 = data.reshape((1,-1))
print data1
print data1.shape
