# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0101_input_au.py
@time: 2016/11/2 12:46
"""

import os
import sunau
import pylab as pl
import numpy as np

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
wave_data = np.fromstring(str_data, dtype=np.short)
time = np.arange(0, nframes) * (1.0 / framerate)

# 绘制波形
pl.subplot(211)
pl.plot(time, wave_data, c="g")
pl.xlabel("time (seconds)")
pl.show()
