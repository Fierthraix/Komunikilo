#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == '__main__':
    FFT_LEN = 8192
    f1 = 100
    f2 = 50
    f3 = 150

    fs = 10_000
    t_step = 1 / fs
    num_samps = FFT_LEN * 5

    time = [i * t_step for i in range(num_samps)]

    s = lambda f: [np.cos(2 * np.pi * f * t) for t in time]
    s1 = s(f1) 
    s2 = s(f2) 
    s3 = s(f3) 

    eps = 20
    channel = [i + j + k for i, j, k in zip(s1, s2, s3)]

    filtered = butter_bandpass_filter(channel, f1 - eps, f1 + eps, fs, order=4)
    plt.plot(time[:5000], s1[:5000])
    # plt.plot(time[:5000], channe[:5000l)
    plt.plot(time[:5000], filtered[:5000])
    plt.show()
        
