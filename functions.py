import numpy as np
import cv2
import numpy.fft as fft
from matplotlib import pyplot as plt
import scipy as sy
import pdb


def trackFeatures(cap, old_frame, corners_t, xx, yy, color, old_gray):
    print "Feature Tracking (Lucas Kanade) ..."
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mask = np.zeros_like(old_frame)
    cnt = 0
    listx = []
    listmain = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break

        cnt = cnt + 1
        if cnt == 301:
            break
        listx.append(cnt)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners_t, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = corners_t[st == 1]

        # draw the tracks
        templist = []

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            x1 = int(xx + a)
            y1 = int(yy + b)
            x2 = int(xx + c)
            y2 = int(yy + d)
            templist.append(float(b))

            cv2.line(mask, (x1, y1), (x2, y2), color[i].tolist(), 2)

        img = cv2.add(frame, mask)
        listmain.append(templist)
        cv2.imshow('frame', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    # pdb.set_trace()
    return (listx, listmain)


def markFeatures(old_frame, focused_face, corners, xx, yy):
    print "Marking features..."
    for i in corners:
        ix, iy = i.ravel()
        cv2.circle(focused_face, (ix, iy), 3, 255, -1)
        cv2.circle(old_frame, (xx + ix, yy + iy), 3, 255, -1)


def getFocusedArea(face, old_frame):
    print "Selecting boundary for features..."
    f = 0
    xx = 0
    yy = 0
    '''
        for (x, y, w, h) in face:
            if f == 0:
                xx = x
                yy = y
                f = 1
            focused_face = old_frame[y:y + h / 4, x: x + (4 * w / 5)] # [y: y+h, x: x+w]#

        for (x,y,w,h) in face:
            if f==0:
                xx=x+w/4
                yy=y
                f=1
            cv2.rectangle(old_frame, (x,y), (x+w,y+h), (0,255,0),2)
            focused_face = old_frame[y:y + 8*h / 10, x + w/4: x + (3 * w / 4)]#[y: y+h, x: x+w]#
            cv2.rectangle(old_frame, (x+w/4,y), (x+3*w/4,y+8*h/10), (255,0,0),2)
            cv2.imshow('img',old_frame)
    '''
    for (x, y, w, h) in face:
        if f == 0:
            xx = x + 5 * w / 12
            yy = y
            f = 1
        cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        focused_face = old_frame[y:y + 6 * h / 10, x + 5 * w / 12: x + (7 * w / 12)]
        cv2.rectangle(old_frame, (x + 5 * w / 12, y), (x + 7 * w / 12, y + 6 * h / 10), (255, 0, 0), 2)

    return (focused_face, xx, yy)


def getShiTomasiCorners(old_frame, focused_face, maxCorners):
    print "Locating feature points..."
    # params for ShiTomasi corner detection EXTENSION OF HARRIS CORNER DETECTION
    feature_params = dict(maxCorners=maxCorners,
                          qualityLevel=0.01,
                          minDistance=1,
                          blockSize=7)

    face_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.cvtColor(focused_face, cv2.COLOR_BGR2GRAY)

    corners_t = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    corners = np.int0(corners_t)

    return (corners_t, corners, face_gray)


def videoCap(loc):
    print "Video Capture..."
    cap = cv2.VideoCapture(loc)
    fps = cap.get(5)
    print ("fps", fps)

    return (cap, fps)


def faceClassifier(cap):
    print "Face Classifier..."
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()

    # cv2.imshow('Old_Frame', old_frame)
    cv2.waitKey(0)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    restart = True
    # while restart == True:
    face = face_classifier.detectMultiScale(old_gray, 1.2, 4)

    if len(face) == 0:
        print "This is empty"

    return old_frame, face, color, old_gray

def fft_plo(data,fs):

    for row in data.T:
        spectrum = fft.fft(row)
        freq = fft.fftfreq(n=len(spectrum),d=1/fs)
        plt.plot(freq, abs(spectrum))
        print ('freq', freq)



    plt.legend(["s[0]", "s[1]", "s[2]", "s[3]", "s[4]"])
    plt.show()

def fft_plot2(data):
    spectrum = fft.fft(data)
    freq = fft.fftfreq(n=len(spectrum))
    plt.plot(freq, abs(spectrum))
    print ('freq', freq)
    plt.show()
    #pdb.set_trace()

def fTransform(data):
    fpulse = []
    #data.shape = nfeatures x nsamples
    for row in data:
        spectrum = sy.fft(row)

        freq = sy.fftpack.fftfreq(len(spectrum))
        #print ("freq", freq) #max(freq), min(freq))
        '''threshold = 0.5 * max(abs(spectrum))
        mask = abs(spectrum) > threshold  # THRESHOLD ???????
        peaks = freq[mask]
        fpulse.append(max(peaks))'''

    return fpulse




































