# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0400_nearest_centroid.py
@time: 2016/11/25 10:29
"""

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid

data = np.loadtxt('data/transAllData.txt')
num = 100 # 每种风格音乐文件个数
tempArray = []
y = []
for genre in range(10): #共10种风格
    for i in range(num):
        y.append(genre)
labels = np.array(y)

data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)

clf = NearestCentroid(metric='euclidean', shrink_threshold=None)
clf.fit(data_train, label_train)
print clf.score(data_test, label_test)