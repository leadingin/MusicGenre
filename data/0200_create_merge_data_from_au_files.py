# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0200_create_merge_data_from_au_files.py
@time: 2016/12/9 5:40
"""

from __future__ import print_function
import data.load_raw_data_to_file as loader
import os

def read_au_file_to_csv(file_path, csv_path):
    # file_path = "F:/dh/DL/music/GTZAN_genres_au"
    # csv_path = "F:/python/Projects/MusicGenre/data/"

    ld = loader.LoadRawData()

    # 读取au文件里的数据存至txt
    #ld.save_raw_data_to_file(file_path)
    # 列转成行
    for root, subdirs, files in os.walk(csv_path + "raw"):
        files.sort()
        for file in files:
            ld.convert_single_files("raw/" + file, "converted/" + file.replace(".txt", "-cvt.txt"))
    # 合并到一个文件
    for root, subdirs, files in os.walk(csv_path + "converted"):
        files.sort()
        for file in files:
            ld.merge_cvt_files("converted/" + file, "merge/allRawData.txt")

    print("Finished.")


def spilit_data(self, training_scale):
    pass

if __name__ == "__main__":
    # 准备所有的原始数据
    read_au_file_to_csv(file_path = "C:/Users/song/data/GTZAN_genres_au", csv_path = "C:/Users/song/PycharmProjects/MusicGenre/data/")
