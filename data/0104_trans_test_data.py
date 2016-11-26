# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0104_trans_test_data.py
@time: 2016/11/23 20:11
"""
import numpy as np

data = np.loadtxt('data/testFeatureData.txt')
# print data
# print data.shape

'''
按照列方向转换，每20列代表一个文件
新的数组内每一行代表一个文件

numpy test
a = np.array([1,2])
b = np.array([2,3])
print np.vstack((a,b))
'''
result = np.array([]) #最终结果
tempArray = [] #存放每个文件的feature
counter = 0
for j in range(data.shape[1]): #每20列代表一个文件
    t = j + 1
    if j != 0 and t % 20 == 0:
        if result.size == 0:
            for i in range(data.shape[0]):  # 行 433
                tempArray.append(np.float32(data[i][j]))
            result = np.array([tempArray])
            tempArray = []
        else:
            for i in range(data.shape[0]):  # 行 433
                tempArray.append(np.float32(data[i][j]))
            result = np.vstack((result, tempArray))
            tempArray = []
    else:
        for i in range(data.shape[0]):  # 行 433
            tempArray.append(np.float32(data[i][j]))

print result.shape

np.savetxt('data/transTestData.txt',result,fmt='%s',newline='\n')





