# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0201_add_class_in_each_row.py
@time: 11/28/16 7:20 PM
"""

from __future__ import print_function

TOTAL_ROW_NUM = 1000

# line 0-999
# line 0-99    class 0
# line 100-199 class 1
# ...
# line 900-999 class 9

class_num = 0
line_num = 0


fr = open("merge/allRawData.txt", "r")
fw = open("merge/raw_data.txt", "w")
'''
fr = open("transAllData.txt", "r")
fw = open("merge/scat_data.txt", "w")
'''
for line in fr:
    # line += str(class_num) not worked
    line = line.strip() + " " + str(class_num) + "\n" # cancel "\n" and "\r"
    fw.write(line)
    line_num += 1
    if line_num % 100 == 0:
        class_num += 1
        print ("%i / 1000 lines finished." % (line_num))
fr.close()
fw.close()

print ("Verify new file:")

'''
fr1 = open("merge/raw_data.txt", "r")
'''
fr1 = open("merge/scat_data.txt", "r")
for i in range(1000):
    line_num1 = i + 1
    l = fr1.readline()[-10:]
    if (line_num1+1) % 100 == 0 or (line_num1-1) % 100 == 0 or line_num1 % 100 == 0:
        print ("line-" + str(line_num1) + ":" + l.strip())
fr1.close()
print ("Finished.")










