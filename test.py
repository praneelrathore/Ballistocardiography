import cPickle
import peakutils
from peakDetection import *

x = cPickle.load(open('s01.dat', 'rb'))
data =  x['data']
array = data[0,38]
print array


#ar1 = array[0:2500]

#ans=pd.testplotFinalBPM(array,3000,15000,250)
#ans = pd.calculateBPM(indexes,250)
#print ans


