# -*- coding:utf-8 -*-

"""
@author: Songgx
@name: 0102_test_cross_validation.py
@time: 2016/11/23 20:03
"""

from sklearn import cross_validation
from sklearn import datasets, svm   #导入所需要的库

k_fold = cross_validation.KFold(n=6, n_folds=3)
for train_indices, test_indices in k_fold:
    print('Train: %s | test: %s' % (train_indices, test_indices))

digits = datasets.load_digits()   #load scikit-learn库里的实验数据集：digits
X_digits = digits.data  #获得样本特征数据部分
y_digits = digits.target  #获得样本label
svc = svm.SVC(C=1, kernel='linear')  #初始化svm分类器
kfold = cross_validation.KFold(len(X_digits), n_folds=3) #初始化交叉验证对象，len(X_digits)指明有多少个样本；n_folds指代kfolds中的参数k,表示把训练集分成k份（n_folds份），本例中为3份
for train, test in kfold:
    print svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
    #此处train、test里有交叉验证对象中已经初始化好的3组训练样本和测试样本所需的位置标号
##其实cross_validation库将上述for循环也集成进来了
#cross_validation.cross_val_score(svc, X_digits, y_digits, n_jobs=-1)  #n_jobs=-1代表将受用计算机上的所有cpu计算,参数cv（此例中为默认值）除了kfold选项，还可以选择StratifiedKFold等,如果cv是一个int数字的话，并且如果提供了raw target参数，那么就代表使用StratifiedKFold分类方式，如果没有提供raw target参数，那么就代表使用KFold分类方式。
