# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0200_svm.py
@time: 2016/11/24 18:06
"""

from __future__ import print_function
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

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

clf = SVC(C=16, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.00024, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
clf.fit(data_train, label_train)
print (clf.score(data_test, label_test))
