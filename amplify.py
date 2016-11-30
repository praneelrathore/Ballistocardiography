from eularian_magnification import base as bs
import cv2
import matplotlib
import scipy
import numpy as np

bs.eulerian_magnification('sample.mp4', image_processing='gaussian', pyramid_levels=3, freq_min=0, freq_max=500.0, amplification=500)
#bs.show_frequencies('sample2_min0_max500.0_amp500_magnified.avi')