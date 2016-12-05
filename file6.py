from matplotlib import pyplot as plt
import numpy as np
import os, re
import matplotlib.mlab as mlab
import functions as func
from scipy.integrate import simps
from scipy.stats import gaussian_kde
import scipy.stats.stats as st

#1,2,3,4,5,7,8,10,15,21,22

subject = '22'

dirpos = func.get_paths()["saved_hrv_positive"]
dirneg = func.get_paths()["saved_hrv_negative"]
hrvpos = np.array(0)
hrvneg = hrvpos

for file in os.listdir(dirpos):
    if re.match('s'+subject, file):
        x = np.genfromtxt(dirpos+file, delimiter=',')
        hrvpos = np.append(hrvpos,x)

for file in os.listdir(dirneg):
    if re.match('s'+subject, file):
        x = np.genfromtxt(dirneg+file, delimiter=',')
        hrvneg = np.append(hrvpos,x)

ske = {-0.314, -0.208, 0.012, -0.027, 0.0756, 0.697, 0.738, 0.889, 1.046, 1.081, -0.595, 0.567, -0.125, -0.037,
           -0.074, -0.052, -0.21, -0.16, 0.327, 0.329, 0.133, 0.292}
new_skw = func.convert19(ske)

print ('new skew', new_skw)
print len(new_skw)

exit(0)

###############################################################################################

plt.subplot(221)
y, bins, patches = plt.hist(x=hrvpos,bins=10)
#bincenters = 0.5*(bins[1:]+bins[:-1])
#plt.plot(bincenters,y,'-')
plt.title("HRV Histogram POS")
#plt.xlabel("HRV")
plt.ylabel("Frequency")

x=hrvpos
density = gaussian_kde(x)
xs = np.linspace(0,max(x),len(x))
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.subplot(222)
plt.plot(xs,density(xs))

stdpos = np.std(x)

print len(ske)


interval=0.3
area = func.integrate(x, interval)#simps(x,0.3)#
c = func.areaInterval(x)
ans = func.convert09(c)
print "The area (POS) is", area
print "The area for intervals POS ", ans
print "SD for hrvPOS ", stdpos
print "skewness POS:", st.skew(x, bias = False)
print "kurtosis POS:", st.kurtosis(x, bias = False)


##########################################################################################################


plt.subplot(223)
y, bins, patches = plt.hist(x=hrvneg,bins=10)
#bincenters = 0.5*(bins[1:]+bins[:-1])
#plt.plot(bincenters,y,'-')
plt.title("HRV Histogram NEG")
plt.xlabel("HRV")
plt.ylabel("Frequency")

x=hrvneg
density = gaussian_kde(x)
xs = np.linspace(0,max(x),len(x))
density.covariance_factor = lambda : .25
density._compute_covariance()
plt.subplot(224)
plt.plot(xs,density(xs))

stdneg = np.std(x)
print

interval=0.3
area = func.integrate(x, interval)#simps(x,0.3)#
c = func.areaInterval(x)
ans = func.convert09(c)
print "The area (NEG) is", area
print "The area for intervals NEG ", ans
print "skewness NEG:", st.skew(x, bias = False)
print "kurtosis NEG:", st.kurtosis(x, bias = False)


########################################################################################################

plt.show()
