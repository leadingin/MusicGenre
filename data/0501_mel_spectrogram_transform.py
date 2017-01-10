# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0501_mel_spectrogram_transform.py
@time: 2017/1/10 10:47
"""

import os
import sunau
import numpy as np
import librosa as lb

# 转换的输出目录
output_path = "C:/Users/song/data/mel_spectrogram/"

# 转换参数
Fs         = 12000#sampling rate
N_FFT      = 512#length of fft window
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 29.12

#hop length:no of samples between successive frames S:power spectrogram y: audio time series


def log_scale_melspectrogram(path):
    signal, sr = lb.load(path, sr=Fs)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]

    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref_power=1.0)

    return melspect

def mel_spectrogram_trans(au_file_path):
    melspect = log_scale_melspectrogram(au_file_path)

    if au_file_path.find("/") != -1:
        au_filename = au_file_path[(au_file_path.rfind("/")+1):]
    else:
        au_filename = au_file_path

    au_filename = au_filename.replace(".au", ".mel_spectrogram")
    np.savetxt(output_path + au_filename, melspect, fmt='%s', delimiter=' ')

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