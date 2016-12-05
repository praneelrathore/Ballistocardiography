from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low,high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs):
    #print('In Butterworth Filter data.shape', data.shape)
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, lowcut, fs):
    #print('In Butterworth Filter data.shape', data.shape)
    b, a = butter_highpass(lowcut, fs)
    y = lfilter(b, a, data)
    return y

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butterworthHighPass(T, nsamples, fs, lowcut, data):
    print "Butterworth BandPass Filter (0.05,0.5) Hz..."
    t = np.linspace(0, T, nsamples, endpoint=False)
    plt.plot(t, data, label='Noisy signal')

    y = butter_highpass_filter(data, lowcut, fs)

    plt.plot(t, y, label='Filtered signal')

    plt.xlabel('frame number')
    plt.legend(['noisy signal', 'filtered signal'])
    plt.show()
    return y

def butterworthBandpass(T, nsamples, fs, lowcut, highcut, data):
    print "Butterworth BandPass Filter (0.75,5) Hz..."
    t = np.linspace(0, T, nsamples, endpoint=False)
    plt.plot(t, data.T[0], label='Noisy signal')

    idata_newt = np.zeros(data.T.shape)
    i = 0
    for column in data.T:
        y = butter_bandpass_filter(column, lowcut, highcut, fs)
        idata_newt[i] = y
        i = i + 1

    idata_new = idata_newt.T

    plt.plot(t, idata_new.T[0], label='Filtered signal')

    plt.xlabel('frame number')
    plt.legend()
    plt.show()
    return idata_new
