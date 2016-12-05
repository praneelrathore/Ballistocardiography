import cv2
import functions as func
import cubicSplineInterpolation as csi
import butterworthFilter as bwf
import pca
import peakDetection as pd
import numpy as np
from matplotlib import pyplot as plt

frCnt = -1

'''
loc = "Aayush.mp4"
maxCorners = 50
frCnt = 195
#BPM = 70, PPG using MI Band = 71, 73, 75 BPM 12 seconds video
'''

video_num={'01'}
subject_num= '01'

for xi in video_num:
    video = 's'+ subject_num + '_trial' + xi
    print ('Working on video ', video)

    loc = func.get_paths()["sample_videos"] + video + '.avi'
    filename = func.get_paths()["saved_signal"] + video + '_signal.out'
    filename_hrv = func.get_paths()["saved_hrv"] + video + '_hrv.out'

    maxCorners = 100
    #frCnt = 501
    #BPM = 72, using PPG data in file5.py, BPM = 71.4


    lowcut = 0.75
    highcut = 5.0
    (cap,fps) = func.videoCap(loc)
    upSampleFactor = float(250.0/fps)
    samplingFreq = float(fps * upSampleFactor)
    print ('fs', samplingFreq)
    print ('upSampleFactor', upSampleFactor)
    thres = 0.25

    # USES HAAR CASCADE TO DETECT FACE IN THE VIDEO
    (old_frame, face, color, old_gray) = func.faceClassifier(cap)

    # SELECTS THE BOUNDARY FOR FEATURE EXTRACTION
    (focused_face,xx,yy) = func.getFocusedArea(face, old_frame)

    # RETURNS THE CORNER POINTS
    (corners_t,corners,face_gray) = func.getShiTomasiCorners(old_frame,focused_face, maxCorners)

    # MARKS THE FEATURES ON FACE
    func.markFeatures(old_frame,focused_face,corners,xx,yy)

    # LUCAS KANADE FEATURE TRACKING
    (listx, listmain,cnt) = func.trackFeatures(cap,old_frame,corners_t,xx,yy, color, old_gray, frCnt)

    # FORMS DATA FROM THE TRACKED FEATURE POINTS
    dataPoints = pca.formData(listmain)
    print ('data.shape', dataPoints.shape)

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
    (bpm,hrv,SDNN) = pd.plotFinalBPM(s[i],T,nsamples,samplingFreq,fpulse,thres,cnt)
    #np.savetxt(filename_hrv, hrv, delimiter=',')
    #np.savetxt(filename, s[i], delimiter=',')

    print ("Final BPM = %f, fpulse = %f, SDNN = %f" %(bpm,fpulse,SDNN))

    cv2.destroyAllWindows()
    cap.release()
    #plt.close('all')