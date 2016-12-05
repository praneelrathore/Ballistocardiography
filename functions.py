import numpy as np
import cv2
import numpy.fft as fft
from matplotlib import pyplot as plt
import scipy as sy
import pdb
import json
import os

def convert09(data):
    old_min = min(data)
    old_max = max(data)
    new_min = 0
    new_max = 9

    arr = np.zeros(5)
    i = 0
    for old_value in data:
        arr[i] = 9.0 - (( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min)
        i=i+1

    return arr

def convert19(data):
    old_min = min(data)
    old_max = max(data)
    new_min = 1
    new_max = 9

    arr = np.zeros(len(data))
    i = 0
    for old_value in data:
        arr[i] = 9 - (( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min)
        i=i+1

    return arr


def areaInterval(data):
    unique, counts = np.unique(data, return_counts=True)
    ar1 = 0
    ar2 = 0
    ar3 = 0
    ar4 = 0
    ar5 = 0
    arr = np.zeros(5)

    #print dict(zip(unique, counts))

    for v, (i,cnt) in enumerate(zip(unique, counts)):
        if (i >= 0.4 and i < 0.7):
            ar1 += cnt
        elif (i >= 0.7 and i < 1.0):
            ar2 += cnt
        elif (i >= 1.0 and i < 1.3):
            ar3 += cnt
        elif (i >= 1.3 and i < 1.6):
            ar4 += cnt
        elif (i >= 1.6 and i < 1.9):
            ar5 += cnt

    arr[0] = ar1
    arr[1] = ar2
    arr[2] = ar3
    arr[3] = ar4
    arr[4] = ar5

    return arr



def integrate(y_vals, h):
    i=1
    total=y_vals[0]+y_vals[-1]
    for y in y_vals[1:-1]:
        if i%2 == 0:
            total+=2*y
        else:
            total+=4*y
        i+=1
    return total*(h/3.0)

def get_paths():
    paths = json.loads(open("SETTINGS.json").read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def compute(list1, list2):
    k=len(list2) - len(list1)
    #for i in range(0, len(list2)):
    i=0
    while(i<len(list2)):
        if (i==len(list1)):
            list2 = list2[0:i-1]
            print 'here1'
            return list2
        elif abs(list1[i] - list2[i]) >= 2.0:
            del list2[i]
            k=k-1
            if k==0:
                print 'here2 compute1'
                return list2
        else:
            i=i+1


def process(listmain):
    for i in range(1, len(listmain)):
        if len(listmain[i - 1]) < len(listmain[i]):
            listmain[i] = compute(listmain[i - 1], listmain[i])
    return listmain

def trackFeatures(cap, old_frame, corners_t, xx, yy, color, old_gray, frCnt):
    print "Feature Tracking (Lucas Kanade) ..."
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                     flags=0)
    mask = np.zeros_like(old_frame)
    cnt = 0
    listx = []
    listmain = []

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == False:
            break

        cnt = cnt + 1
        if frCnt != -1:
            if cnt == frCnt:
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
    return (listx, listmain, cnt-1)


def markFeatures(old_frame, focused_face, corners, xx, yy):
    print "Marking features..."
    for i in corners:
        ix, iy = i.ravel()
        cv2.circle(focused_face, (ix, iy), 3, 255, -1)
        cv2.circle(old_frame, (xx + ix, yy + iy), 3, 255, -1)

    #plt.imshow(old_frame), plt.show()

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
        #cv2.imshow('img', old_frame)


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
    fs = 1/128.0
    freq = fft.fftfreq(n=len(spectrum),d=float(fs))
    plt.plot(freq, abs(spectrum))
    #plt.plot(freq)
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




































