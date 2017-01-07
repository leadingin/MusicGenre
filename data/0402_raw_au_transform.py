# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0402_wavelet_transform.py
@time: 2017/1/7 13:30
"""

# (661794,)
# 131072 大约6s的子采样
# 131072 * 5 = 655360
# 截断最后 6434个采样点 661794-655360=6434

import os
import sunau
import numpy as np

# 转换的输出目录
output_path = "raw_data/"

def wavelet_trans(au_file_path, size=131072):
    music = sunau.open(au_file_path, 'r')

    if au_file_path.find("/") != -1:
        au_filename = au_file_path[(au_file_path.rfind("/")+1):]
    else:
        au_filename = au_file_path

    nframes = music.getnframes()

    # 读取波形数据
    str_data = music.readframes(nframes)
    music.close()

    # 将波形数据转换为数组
    wave_data = np.fromstring(str_data, dtype=np.short)

    # wavelet transform, db6

    # 分割的数量spilit_num
    spilit_num = int((wave_data.shape[0]) / size)
    for i in range(spilit_num):
        spilit_sample = wave_data[size * i:size * (i+1)]
        spilit_sample_name = (au_filename + "_" + str(i)).replace(".au", ".raw")

        np.savetxt(output_path + spilit_sample_name, spilit_sample, fmt='%s', delimiter=' ')

    print("{} Done.".format(au_filename))


def transform_all_files(folder_path):
    for root, subdirs, files in os.walk(folder_path):
        subdirs.sort()
        # ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        # 分别对应[0-9]
        for subdir in subdirs:
            path = root + "/" + subdir
            for root1, subdirs1, files1 in os.walk(path):
                for file1 in files1:
                    cur_path = root1 + "/" + file1
                    wavelet_trans(cur_path)


if __name__ == "__main__":
   #wavelet_trans('classical.00000.au') # 检查单个转换无误过后进行全部转换
   transform_all_files("D:/dh/DL/music/genres")