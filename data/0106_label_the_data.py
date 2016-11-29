# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0106_label_the_data.py
@time: 2016/11/24 16:53


打散数据
http://friskit.me/2014/10/22/shuffle-train-data-in-numpy/
"""

import numpy as np

data = np.loadtxt('transAllData.txt')
num = 100 # 每种风格音乐文件个数
tempArray = []
y = []
for genre in range(10): #共10种风格
    for i in range(num):
        y.append(genre)
labels = np.array(y)
print labels


