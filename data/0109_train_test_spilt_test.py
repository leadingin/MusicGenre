# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0109_train_test_spilt_test.py
@time: 2016/11/24 17:23
"""

from sklearn.cross_validation import train_test_split
import numpy as np
data = np.reshape(np.random.randn(20),(10,2)) # 10 training examples
labels = np.random.randint(2, size=10) # 10 labels
x1, x2, y1, y2 = train_test_split(data, labels, test_size=0.2)

print data
print labels

print "#################"

print x1
print x2
print y1
print y2