from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter, lfilter
import numpy.fft as fft
from sklearn.decomposition import PCA
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture('s01_trial01.avi')

fps = cap.get(5)
print "fps"
print fps  # FPS 24.083333

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.0001,
                      minDistance=10,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100000, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

#  cv2.imshow('Old_Frame', old_frame)
cv2.waitKey(0)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
restart = True
# while restart == True:
face = face_classifier.detectMultiScale(old_gray, 1.2, 4)

if len(face) == 0:
    print "This is empty"
f = 0
xx = 0
yy = 0
for (x, y, w, h) in face:
    if f == 0:
        xx = x + 5 * w / 12
        yy = y
        f = 1
    cv2.rectangle(old_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    focused_face = old_frame[y:y + 6 * h / 10, x + 5 * w / 12: x + (7 * w / 12)]  # [y: y+h, x: x+w]#
    cv2.rectangle(old_frame, (x + 5 * w / 12, y), (x + 7 * w / 12, y + 6 * h / 10), (255, 0, 0), 2)
    cv2.imshow('img', old_frame)

# face_gray = cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(focused_face, cv2.COLOR_BGR2GRAY)

corners_t = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
corners = np.int0(corners_t)

for i in corners:
    ix, iy = i.ravel()
    cv2.circle(focused_face, (ix, iy), 3, 255, -1)
    cv2.circle(old_frame, (xx + ix, yy + iy), 3, 255, -1)

cv2.imshow('img', old_frame)
mask = np.zeros_like(old_frame)
listx = []
cnt = 0
listmain = []
f = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    cnt += 1
    listx.append(cnt)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, corners_t, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = corners_t[st == 1]

    # draw the tracks
    # print "COLORING TIME!"

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
    # print templist
    listmain.append(templist)

    # listy.append(b)
    cv2.imshow('frame', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

k = 100
for list in listmain:
    k = min(len(list), k)

for list in listmain:
    if len(list) > k:
        l = len(list)
        list[k:l] = []

np_array = np.asarray(listmain)

##### PLOTTING AND CUBIC-SPLINE INTERPOLATION #####


nsamples = 10 * len(listx)
# plt.plot(listx, np_array.T[0], 'r')
np_array_new = np.zeros((len(np_array[0]), nsamples))
i = 0
xvals = np.linspace(min(listx), max(listx), nsamples)
for column in np_array.T:
    tck = interpolate.splrep(listx, column)
    yvals = interpolate.splev(xvals, tck)
    np_array_new[i] = yvals
    i = i + 1

idata = np_array_new.T
# print np_array_new.shape
# print idata.shape
yval = idata.T[0]
plt.plot(xvals, yval, 'b')

plt.legend(['Linear', 'Cubic Spline'])
plt.xlabel('frame number')
plt.ylabel('feature y position')
plt.show()


#####   BUTTERWORTH BANDPASS FILTER BETWEEN 0.75, 5 HZ   #####







def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    return y


fs = fps * 5  #SAMPLING FREQUENCY 30/150
lowcut = 0.75  # EVERYTHING IN HZ
highcut = 5

T = max(listx)
t = np.linspace(0, T, nsamples, endpoint=False)
plt.plot(t, yval, label='Noisy signal')

idata_newt = np.zeros(idata.T.shape)
i = 0
for column in idata.T:
    y = butter_bandpass_filter(column, lowcut, highcut, fs)
    idata_newt[i] = y
    i = i + 1

idata_new = idata_newt.T
# print "idata_new.shape"
# print idata_new.shape  #(10*frames,27)
plt.plot(t, idata_new.T[0], label='Filtered signal')

plt.xlabel('frame number')
plt.legend()
plt.show()

#####   PRINCIPLE COMPONENT ANALYSIS   #####
print
print
print "PCA..."
pca = PCA()
X = idata_new

X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / nsamples
eigenvalues = pca.explained_variance_

i = 0
s = np.zeros((5, nsamples))
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
    s[i] = np.dot(idata_new, eigenvector)
    print ("s[i].shape", s[i].shape)
    i += 1
    if i == 5:
        break

plt.plot(t, s[0], 'r', t, s[1], 'b', t, s[2], 'g', t, s[3], 'c', t, s[4], 'y')
plt.legend(["s[0]", "s[1]", "s[2]", "s[3]", "s[4]"])
plt.show()
print ("s.shape", s.shape)

fpulse = -100
for row in s:
    spectrum = fft.fft(row)
    freq = fft.fftfreq(len(spectrum))
    plt.plot(freq, abs(spectrum))
    threshold = 0.25 * max(abs(spectrum))
    mask = abs(spectrum) > threshold  # THRESHOLD ???????
    peaks = freq[mask]
    # print ("peaks", peaks)
    p1 = max(peaks)
    fpulse = max(p1, fpulse)

plt.legend(["s[0]", "s[1]", "s[2]", "s[3]", "s[4]"])
plt.show()
print ("fpulse", fpulse)

bpm = float(60 / fpulse)
print ("bpm", bpm)

cv2.destroyAllWindows()
cap.release()
