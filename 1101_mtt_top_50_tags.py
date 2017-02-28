# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1101_mtt_top_50_tags.py
@time: 2017/1/15 13:52
"""

import numpy as np


# 返回特定title的序号，在tfrecords里，多一项map3_id，所以+1
def index_of_the_title(title_arr):
    index_arr=[]
    for i in range (len(title_arr)):
        index_arr.append(i+1)
    return index_arr


mtt_tags = np.loadtxt('data/MTT_tags.csv', delimiter=',', skiprows=0, dtype=str)
#print(mtt_tags.shape)
print('Tag file loaded.')

title_arr=[]
for elemt in mtt_tags[0][1:-1]:
    title_arr.append(str(elemt)[2:-1])


val_array=[]

val_arr = mtt_tags[1:]
val_arr_ = val_arr.tolist()
for row in val_arr:
    row = row[1:-1]
    temp_row=[]
    for e in row:
        temp_row.append(int(str(e)[2:-1]))
    val_array.append(temp_row)

#print(len(title_arr)) # 188
#print(len(val_array)) # 188

result_arr = []
for _ in range(188):
    result_arr.append(0)

for row in val_array:
    for i in range(len(result_arr)):
        result_arr[i] += row[i]


result_dict = dict(zip(title_arr, result_arr))
# print(result_dict)


'''
按照value的值从大到小的顺序来排序。

dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
dict= sorted(dic.items(), key=lambda d:d[1], reverse = True)
print(dict)


对字典按键（key）排序：

dic = {'a':31, 'bc':5, 'c':3, 'asd':4, 'aa':74, 'd':0}
dict= sorted(dic.items(), key=lambda d:d[0])
print dict
'''

# 排序后的dict
dictionary= sorted(result_dict.items(), key=lambda d:d[1], reverse = True)

dict_top_50={}
for i in range(50):
    dict_top_50.update({dictionary[i]})

dict_top_50_sorted = sorted(dict_top_50.items(), key=lambda d:d[1], reverse = True)
print(dict_top_50_sorted)
# print(len(dict_top_50_sorted))  # 50

title_index_dict = dict(zip(title_arr, index_of_the_title(title_arr)))

index_top_50 = []

keys = dict_top_50.keys()
for key in keys:
    index_top_50.append(title_index_dict[key])

print(index_top_50)

top_50_tags_np_arr = np.array(index_top_50)
output_path='data/top_50_tags.txt'
np.savetxt(output_path, top_50_tags_np_arr, fmt='%s', delimiter=',')
print("Top 50 tags' index saved in {}".format(output_path))









