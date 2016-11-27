# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0000_preparing_data.py
@time: 2016/11/27 12:20
"""

import data.load_raw_data_to_file as loader
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