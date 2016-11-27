import cPickle
import numpy as np
from matplotlib import pyplot as plt
import butterworthFilter as bwf
import  functions as func
import peakDetection as pd

lowcut = 0.2
highcut = 2
samplingFreq = 128 #Hz
fpulse = 1.3

x = cPickle.load(open('s01.dat', 'rb'))
testdata = x['data']
testarray = testdata[0,38]
print ('testarray.shape',testarray.shape)

data = np.array(testarray)
plt.plot(data)
plt.show()

T = len(data)
nsamples = len(data)

func.fft_plot2(data)

#y = bwf.butterworthBandpass2(T,nsamples,samplingFreq,lowcut,highcut,data)
#func.fft_plot2(y)

bpm = pd.plotFinalBPM(data,T,nsamples,samplingFreq,fpulse)
print ("Final BPM = %f, fpulse = %f" %(bpm,fpulse)) #BPM = 82



