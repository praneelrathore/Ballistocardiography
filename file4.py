import cv2
import functions as func
import cubicSplineInterpolation as csi
import butterworthFilter as bwf
import pca
import peakDetection as pd


loc = "s01_trial01.avi"
maxCorners = 100
upSampleFactor = 5
lowcut = 0.75
highcut = 5.0
(cap,fps) = func.videoCap(loc)
samplingFreq = fps * upSampleFactor

# USES HAAR CASCADE TO DETECT FACE IN THE VIDEO
(old_frame, face, color, old_gray) = func.faceClassifier(cap)

# SELECTS THE BOUNDARY FOR FEATURE EXTRACTION
(focused_face,xx,yy) = func.getFocusedArea(face, old_frame)

# RETURNS THE CORNER POINTS
(corners_t,corners,face_gray) = func.getShiTomasiCorners(old_frame,focused_face, maxCorners)

# MARKS THE FEATURES ON FACE
func.markFeatures(old_frame,focused_face,corners,xx,yy)

# LUCAS KANADE FEATURE TRACKING
(listx, listmain) = func.trackFeatures(cap,old_frame,corners_t,xx,yy, color, old_gray)

# FORMS DATA FROM THE TRACKED FEATURE POINTS
dataPoints = pca.formData(listmain)

# PLOTTING AND CUBIC-SPLINE INTERPOLATION
nsamples = upSampleFactor * len(listx)
nfeatures = len(dataPoints[0])
idata = csi.cubicSplineInterpolate(nsamples,dataPoints,listx, upSampleFactor)

# BUTTERWORTH BANDPASS FILTER BETWEEN 0.75, 5 HZ
T=len(listx)
idata_new = bwf.butterworthBandpass(T, nsamples, samplingFreq, lowcut, highcut, idata)

# PRINCIPAL COMPONENT ANALYSIS - CALCULATION OF S
s = pca.pcaToSignal(idata_new, nsamples, T)  #s.shape = (5,nsamples)

# FOURIER TRANSFORM OF S TO FIND FPULSE
#fpulse = func.fft_plot(s,samplingFreq)

#bpm_s = pd.getBPMforAllSignalComponents(s,T,nsamples,samplingFreq)

(freqComp_s,freqs) = pca.getComponentWithMaxFreqComponent(s,samplingFreq)

i = pca.selectBestPCAComponent(freqComp_s)
fpulse = freqs[i]
print("fpulse = %f" %fpulse)
bpm = pd.plotFinalBPM(s[i],T,nsamples,samplingFreq,fpulse)

print ("Final BPM = %f, fpulse = %f" %(bpm,fpulse))

cv2.destroyAllWindows()
cap.release()