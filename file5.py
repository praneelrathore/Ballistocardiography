import cPickle
import numpy as np
from matplotlib import pyplot as plt
import butterworthFilter as bwf
import  functions as func
import peakDetection as pd
import pca

lowcut = 0.1
highcut = 0.5
samplingFreq = 128.0 #Hz
fpulse = 1.3333
thres = 0.48  #for s01_trial05 testdata[4,38] BPM = 71.4

x = cPickle.load(open('s01.dat', 'rb'))
testdata = x['data']
testarray = testdata[4,38]
print ('testarray.shape',testarray.shape)

data = np.array(testarray)
plt.plot(data)
plt.show()

T = len(data)
nsamples = len(data)

#func.fft_plot2(data)

y = bwf.butterworthHighPass(T,nsamples,samplingFreq,lowcut,data)
#fpulse = pca.getfpulse(np.array(y),samplingFreq)
#func.fft_plot2(y)

bpm = pd.plotFinalBPM(y,T,nsamples,samplingFreq,fpulse,thres)
print ("Final BPM = %f, fpulse = %f" %(bpm,fpulse)) #BPM = 49.6



