import numpy as np
from sklearn.decomposition import PCA
from scipy import signal
import numpy.fft as fft
from matplotlib import pyplot as plt
import math



def normalizeData(data):
    for row in data.T:
        row[:] = [x - row[0] for x in row]
    return data


def formData(listmain, k=100):
    prevlist = []
    for i in range(0,len(listmain)):
        if (len(listmain[i]) == 0):
            listmain[i] = prevlist
        k = min(len(listmain[i]), k)
        prevlist = listmain[i]

    for list in listmain:
        if len(list) > k:
            l = len(list)
            list[k:l] = []

    np_array = np.asarray(listmain)
    np_array = normalizeData(np_array)
    return np_array

def pcaToSignal(X, nsamples, T):
    print "PCA..."
    pca = PCA()
    t = np.linspace(0, T, nsamples, endpoint=False)
    X_transformed = pca.fit_transform(X)

    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.dot(X_centered.T, X_centered) / nsamples
    eigenvalues = pca.explained_variance_

    i = 0
    s = np.zeros((5, nsamples))
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):  # loop runs (no. of times) = num of features
        s[i] = np.dot(X, eigenvector)
        i = i + 1
        if i == 5:
            break

    print ("s.shape", s.shape)

    for row in s:
        plt.plot(row, label='s[%d]' %i)

    plt.legend()
    plt.xlabel("samples")
    plt.ylabel("trajectory")
    plt.show()
    return s

def selectBestPCAComponent(freqComp_s):
    i = np.argmax(freqComp_s)
    return i

def computeMaxFreqComponent(x, fs):
    f, Pxx_den = signal.periodogram(x, fs)
    PSD = sum(Pxx_den)  # np.sqrt(Pxx_den.max())
    i = np.argmax(Pxx_den)
    frac = Pxx_den[i] / PSD
    if (math.isnan(frac)):
        frac=0.0
    return (frac, f[i])


def getComponentWithMaxFreqComponent(s, samplingFreq):
    freqComp_s = []
    templist=[]
    for i in range(0, 5):
        dataset = s[i]
        (freqCom,k) = computeMaxFreqComponent(dataset, samplingFreq)
        print ("Percentage of total PSD accounted for by the freq with max power for s[%d] = %f" % (i, freqCom))
        freqComp_s.append(freqCom)
        templist.append(k)
    print
    return (freqComp_s,templist)

def getfpulse(data, samplingFreq):
    f, Pxx_den = signal.periodogram(data, samplingFreq)
    i = np.argmax(Pxx_den)
    print ('fpulse = ', f[i])

    return f[i]