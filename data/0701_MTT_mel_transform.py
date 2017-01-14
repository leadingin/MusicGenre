# -*- coding:utf-8 -*-

"""
@author: Songgx
@file: 0701_MTT_mel_transform.py
@time: 2017/1/14 14:43
"""

import librosa as lb
import numpy as np
import os


# MTT_wave目录
MTT_wav_path = 'D:/dh/DL/music/MTT_wav'
# 转换的输出目录
output_path = 'C:/Users/song/data/mtt_mel/'

Fs         = 12000#sampling rate
N_FFT      = 512#length of fft window
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 29.12


def mel_spectrogram_trans(wav_file_path):
    melspect = log_scale_melspectrogram(wav_file_path)

    output_path_subfolder = wav_file_path[:(wav_file_path.rfind("/") + 1)]
    output_path_subfolder = output_path_subfolder[-2:] # a/  0/   9/

    if wav_file_path.find("/") != -1:
        wav_filename = wav_file_path[(wav_file_path.rfind("/") + 1):]
    else:
        wav_filename = wav_file_path

    wav_filename = wav_filename.replace(".wav", ".mel_spectrogram")
    np.savetxt(output_path + output_path_subfolder + wav_filename, melspect, fmt='%s', delimiter=' ')

    print("{} Done.".format(output_path_subfolder + wav_filename))


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

def transform_all_files(folder_path):
    for root, subdirs, files in os.walk(folder_path):
        subdirs.sort()
        for subdir in subdirs:
            path = root + "/" + subdir
            for root1, subdirs1, files1 in os.walk(path):
                for file1 in files1:
                    cur_path = root1 + "/" + file1
                    mel_spectrogram_trans(cur_path)

if __name__ == "__main__":
    #mel_spectrogram_trans('D:/dh/DL/music/MTT_wav/0/american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-0-29.wav') # 检查单个转换无误过后进行全部转换
   transform_all_files(MTT_wav_path)

