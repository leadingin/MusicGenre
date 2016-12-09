# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: create_test_data.py
@time: 12/1/16 3:48 PM
"""

'''
f = open("merge/raw_data.txt", "r")
fw = open("merge/raw_data_test.txt", "w")
'''

f = open("merge/scat_data.txt", "r")
fw = open("merge/scat_data_test.txt", "w")


for i in range(1000):
    l = f.readline()
    if i % 100 == 0:
        fw.write(l)

f.close()
fw.close()