# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0601_mfcc_transform.py
@time: 2017/1/10 12:27
"""

# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0501_mel_spectrogram_transform.py
@time: 2017/1/10 10:47
"""

import os
import numpy as np
import librosa as lb

# 转换的输出目录
output_path = "C:/Users/song/data/mfcc/"

# 转换参数
Fs         = 12000#sampling rate
N_FFT      = 512#length of fft window
N_MFCC     = 30
DURA       = 29.12

#hop length:no of samples between successive frames S:power spectrogram y: audio time series


def log_scale_mfcc(path):
    signal, sr = lb.load(path, sr=Fs)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]

    mfcc = lb.logamplitude(lb.feature.mfcc(y=signal, sr=Fs, n_mfcc=N_MFCC) ** 2, ref_power=1.0)
    return mfcc

def mel_spectrogram_trans(au_file_path):
    mfcc = log_scale_mfcc(au_file_path)

    if au_file_path.find("/") != -1:
        au_filename = au_file_path[(au_file_path.rfind("/")+1):]
    else:
        au_filename = au_file_path

    au_filename = au_filename.replace(".au", ".mfcc")
    np.savetxt(output_path + au_filename, mfcc, fmt='%s', delimiter=' ')

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
                    mel_spectrogram_trans(cur_path)


if __name__ == "__main__":
    #mel_spectrogram_trans('classical.00000.au') # 检查单个转换无误过后进行全部转换
   transform_all_files("D:/dh/DL/music/genres")