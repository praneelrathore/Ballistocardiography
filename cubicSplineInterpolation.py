from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate

def cubicSplineInterpolate(nsamples, data, listx, upSampleFactor):
    print "Cubic Spline Interpolation..."
    plt.plot(listx, data.T[0], 'r')
    np_array_new = np.zeros((len(data[0]), nsamples))
    i = 0
    [xvals, sampleSpace] = np.linspace(min(listx), len(listx), nsamples, retstep=True)
    for column in data.T:
        tck = interpolate.splrep(listx, column)
        yvals = interpolate.splev(xvals, tck)
        np_array_new[i] = yvals
        i = i + 1

    idata = np_array_new.T
    plt.plot(xvals, idata.T[0], 'b')

    plt.legend(['Linear', 'Cubic Spline'])
    plt.xlabel('frame number')
    plt.ylabel('feature y position')
    plt.show()
    # pdb.set_trace()
    return idata