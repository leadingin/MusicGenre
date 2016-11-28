# -*- coding:utf-8 -*-

f = open("data/merge/allRawData.txt")
line_num = 0
for line in f:
	line_num += 1
print "File has %i lines" % (line_num)

# 本机win7中有1000行数据
