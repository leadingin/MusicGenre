# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0300_wavelet_hw.py
@time: 2016/12/20 14:23
"""

import os
import sunau
import numpy as np
import pywt

file = os.path.abspath('classical.00000.au')
music = sunau.open(file, 'r')

# 读取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
nchannels = music.getnchannels() # nchannels = 1
sampwidth = music.getsampwidth()
framerate = music.getframerate()
nframes = music.getnframes()

# 读取波形数据
str_data = music.readframes(nframes)
music.close()

#将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.float32)
print (wave_data.shape)
# (330897,)

# wavelet transform, db6
# 单尺度低频系数cA， 单尺度高频系数cD
cA, cD = pywt.dwt(wave_data, 'db6')

cA = np.divide(cA, pow(10, 30))
print(cA.shape)
print(cD.shape)
# (165454,)
# (165454,)
print(cA)

# 32768 大约6s的子采样
# 32768 * 5 = 163840