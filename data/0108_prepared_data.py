# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0108_prepared_data.py
@time: 2016/11/24 17:16
"""
import numpy as np
from sklearn.cross_validation import train_test_split

data = np.loadtxt('data/transAllData.txt')
num = 100 # 每种风格音乐文件个数
tempArray = []
y = []
for genre in range(10): #共10种风格
    for i in range(num):
        y.append(genre)
labels = np.array(y)
'''
indices = np.random.permutation(labels.shape[0]) # 生成随机数列

rand_data_X = data[indices]
rand_data_y = labels[indices]
'''

data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)