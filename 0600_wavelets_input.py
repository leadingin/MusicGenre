# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0600_wavelets_input.py
@time: 2016/12/20 16:41
"""

import numpy as np

cA = np.loadtxt("C:/Users/song/data/wavelets/classical.00000.wavelet_0_cA", dtype=np.float32, delimiter="\n")
cD = np.loadtxt("C:/Users/song/data/wavelets/classical.00000.wavelet_0_cD", dtype=np.float32, delimiter="\n")

wavelets = np.vstack((cA, cD))

print (wavelets.shape)