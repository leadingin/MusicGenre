# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0500_mel_spectrogram_hw.py
@time: 2017/1/9
"""


import numpy as np
import librosa as lb


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

if __name__ == '__main__':
    mfcc = log_scale_mfcc("classical.00000.au")
    print(mfcc.shape)
    print(mfcc)