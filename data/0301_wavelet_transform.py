# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0301_wavelet_transform.py
@time: 2016/12/20 15:15
"""

# (330897,)
# 32768 大约6s的子采样
# 32768 * 5 = 163840

import os
import sunau
import numpy as np
import pywt


output_path = "wavelets/"

def wavelet_trans(au_file_path, size=32768):
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
    wave_data = np.fromstring(str_data, dtype=np.float32)

    # wavelet transform, db6
    # 单尺度低频系数cA， 单尺度高频系数cD
    cA, cD = pywt.dwt(wave_data, 'db6')
    cA = np.divide(cA, pow(10, 30))
    cD = np.divide(cD, pow(10, 30))
    spilit_num = int((cA.shape[0]) / size)
    for i in range(spilit_num):
        cA_i = cA[size * i:size * (i+1)]
        cD_i = cD[size * i:size * (i+1)]
        spilit_sample_name = (au_filename + "_" + str(i)).replace(".au", ".wavelet")

        cA_i_temp_path = spilit_sample_name+"_cA"
        cD_i_temp_path = spilit_sample_name + "_cD"
        np.savetxt(output_path + cA_i_temp_path, cA_i, fmt='%s', delimiter=' ')
        np.savetxt(output_path + cD_i_temp_path, cD_i, fmt='%s', delimiter=' ')

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
   # wavelet_trans('classical.00000.au')
    transform_all_files("D:/dh/DL/music/genres")