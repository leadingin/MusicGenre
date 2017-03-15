# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 1100_roc_auc_hw.py
@time: 2017/1/14 16:48
"""

import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([1., 1., 0., 0.])
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
y_scores = np.array([0.9, 0.9, 0.1, 0.1])
print(roc_auc_score(y_true, y_scores, average='samples'))