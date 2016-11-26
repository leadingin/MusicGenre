# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0100_rawdata_svm.py
@time: 2016/11/25 12:43
"""

import data.load_raw_data_to_file as loader
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
import os


file_path = "F:/dh/DL/music/GTZAN_genres_au"
data_path = "F:/python/Projects/MusicGenre/data/"
#file_path = "/home/wang/sgx/data/genres"

ld = loader.LoadRawData()

# 读取au文件里的数据存至txt
ld.save_raw_data_to_file(file_path)
# 列转成行
for root, subdirs, files in os.walk(data_path+"raw"):
    files.sort()
    for file in files:
        ld.convert_single_files("data/raw/"+file, "data/converted/"+file.replace(".txt", "-cvt.txt"))
# 合并到一个文件
for root, subdirs, files in os.walk(data_path+"converted"):
    files.sort()
    for file in files:
        ld.merge_cvt_files("data/converted/"+file, "data/merge/allRawData.txt")

print "Finished."

'''
num = 100 # 每种风格音乐文件个数
tempArray = []
y = []
for genre in range(10): #共10种风格
    for i in range(num):
        y.append(genre)
labels = np.array(y)

data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.2)

clf = SVC(C=16, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.00024, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)
clf.fit(data_train, label_train)
print clf.score(data_test, label_test)
'''
