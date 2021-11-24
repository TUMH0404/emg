import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

## ローパスフィルタ
def lpf(wave, fs, fe1):
    '''
    Parameter
    ----------
    wave:array-like  1Dのデータ
    fs:サンプリング周波数
    fe1:カットオフ周波数

    '''
    nyq = fs / 2.0
    b, a = signal.butter(4, fe1/nyq, btype='low')
    wave = signal.filtfilt(b, a, wave)
    return wave

# バンドパスフィルタ
def bpf(wave, fs, fe1, fe2):
    '''
    Parameter
    ----------
    fe1、fe2を通す
    wave:array-like  1Dのデータ
    fs:サンプリング周波数
    fe1,fe2:カットオフ周波数

    '''
    nyq = fs / 2.0
    b, a = signal.butter(4, [fe1/nyq, fe2/nyq], btype='band')
    wave = signal.filtfilt(b, a, wave)
    return wave


## 積分値
def integral(x, s0, startindex, endindex, tau=0.001):
    '''
    Return 被積分データ配列xの台形則にもとずく積分データ配列.

    Parameters
    ----------
    x : array-like
        被積分データ. 1-D numpy配列推奨.
    s0 : int or float
        初期値(積分定数).
    statindex : int
        被積分データ配列xの最初のindex.
    endindex : int
        被積分データ配列xの最後のindex.
    tau : float, optional (0.001)
        積分区切り幅.
    '''
    s = s0
    s_arr = [s0]
    for i in range(startindex, endindex+1):
        s += (x[i] + x[i+1]) / 2 * tau
        s_arr.append(s)
    return np.array(s_arr)