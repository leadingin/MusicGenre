# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0500_mel_spectrogram_hw.py
@time: 2017/1/9
"""

import numpy as np
import librosa as lb

from scipy import misc

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

    melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS), ref_power=1.0)

    return melspect

if __name__ == '__main__':
    melspect = log_scale_melspectrogram('jazz.00000.au')
    #melspect = log_scale_melspectrogram('D:/dh/DL/music/MTT_wav/0/american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-0-29.wav')
    print(melspect.shape)
    print(melspect)