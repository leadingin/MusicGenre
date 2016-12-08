# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0300_multi-layer_perceptron.py
@time: 2016/11/24 20:35
"""

from __future__ import print_function
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import train_test_split

data = np.loadtxt('data/transAllData.txt')
num = 100 # 每种风格音乐文件个数
tempArray = []
y = []
for genre in range(10): #共10种风格
    for i in range(num):
        y.append(genre)
labels = np.array(y)

data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)

clf = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
                   beta_1=0.9, beta_2=0.999, early_stopping=False,
                   epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
                   learning_rate_init=0.001, max_iter=200, momentum=0.9,
                   nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                   solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
                   warm_start=False)
clf.fit(data_train, label_train)
print (clf.score(data_test, label_test))