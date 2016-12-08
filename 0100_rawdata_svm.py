# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0100_rawdata_svm.py
@time: 2016/11/25 12:43
"""

from __future__ import print_function
import data.load_raw_data_file_to_array as f2a
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

import time


num = 100 # 每种风格音乐文件个数
tempArray = []
y = []
for genre in range(10): #共10种风格
    # ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    # 分别对应[0-9]
    for i in range(num):
        y.append(genre)
labels = np.array(y)
t1 = time.time()
data = f2a.LoadRawDataFileToArray().load("data/merge/allRawData.txt")
t2 = time.time()
print ("Time cost: %f s." %(t2-t1))
print ("start training")
t3 = time.time()
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)

clf = SVC(C=16, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.00024, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
clf.fit(data_train, label_train)
print (clf.score(data_test, label_test))
t4 = time.time()
print ("Time cost: %f s." %(t4-t3))
