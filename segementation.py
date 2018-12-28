# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Sun Dec 11 2018 11:38:16

@author: aruroxx31
"""

import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL.Image import Image
from skimage import io, transform
from skimage import data
from skimage import color
from skimage.util import view_as_blocks
from scipy import fftpack
from scipy import misc
from skimage.measure import block_reduce
import math
import sys

"""
Segementation Algorithm will be performed on MICC-F220 dataset
"""

%matplotlib inline

img = mpimg.imread('CRW_4810_scale.jpg')     

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


"""
Converted it into GrayScale Image
"""
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
gray = np.array(gray)
gray.shape

"""
Canny Edge Detector
TODO TASK: Implement Without CV2 Libraries
"""

imgcv = cv2.imread('CRW_4810_scale.jpg',0)
edges = cv2.Canny(imgcv,100,200)
plt.subplot(121),plt.imshow(imgcv,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


"""
SLIC SuperPixel Algorithm
"""

rgb = io.imread('CRW_4810_scale.jpg')
lab = color.rgb2lab(rgb)

"""
One Cluster can be represented with 5 parameters
x, y, l, a, b
"""


num_clusters = 200
height = gray.shape[0]
width = gray.shape[1]
numpixels = height * width
s_interval = int(math.sqrt(numpixels/num_clusters))

"""
Initializing in Making Clusters as a list represented by a
cluster center
First covering the width of one row and then the other
"""
clusters = []

h = int(s_interval/2)
w = int(s_interval/2)
while h<height:
    while w<width:
        clusters.append([h, w, rgb[h][w][0], rgb[h][w][1], rgb[h][w][2]])
        print(clusters)
        w = w + s_interval
    w = int(s_interval/2)
    h = h + s_interval


"""
Gradient According to the Paper
"""
def get_gradient(h, w):
    if w + 1 >= width:
        w = width - 2
    if h + 1 >= height:
        h = height - 2

    gradient = rgb[w + 1][h + 1][0] - rgb[w][h][0] + \
               rgb[w + 1][h + 1][1] - rgb[w][h][1] + \
               rgb[w + 1][h + 1][2] - rgb[w][h][2]
    return gradient














