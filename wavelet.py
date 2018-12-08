#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 17:17:56 2018

@author: aruroxx31 and palak
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL.Image import Image
import matplotlib.image as mpimg
from skimage import io, transform
from skimage import data
from skimage import color
from skimage.util import view_as_blocks
import math
import pywt
pywt.Wavelet('Haar')

%matplotlib inline

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


img = mpimg.imread('test.jpg')
plt.imshow(img)


ycbcr = rgb2ycbcr(img)
ycbcr = ycbcr[0: 352, 0: 352]
plt.imshow(ycbcr)
ycbcr = np.array(ycbcr)


"""
Converting 3 Dimensional Ycbcr array into three 2D arrays
"""

yplane = []
cbplane = []
crplane = []

yplane.append(ycbcr[0: , 0:, 0])
cbplane.append(ycbcr[0: , 0:, 1])
crplane.append(ycbcr[0: , 0:, 2])

yplane = np.asarray(yplane)
cbplane = np.asarray(cbplane)
crplane = np.asarray(crplane)

yplane = yplane.reshape(352, 352)
cbplane = cbplane.reshape(352, 352)
crplane = crplane.reshape(352, 352)

plt.imshow(yplane)
plt.imshow(cbplane)
plt.imshow(crplane)

"""
We are going to apply wavelet transform on y plane
"""

import math

"""
Function to write haarTransform to get haarfinal matrix
"""

def HaarWaveletTransform(x):
    N = 8
    row = int(N/2)
    matrix = [0.0]*N

    while True:
        for i in range(0,row):
            sum_val = (x[i * 2] + x[i * 2 + 1])/2
            diff = (x[i * 2] - x[i * 2 + 1])/2
            matrix[i] = sum_val
            matrix[row + i] = diff

        if row == 1:
            return matrix

        x = matrix[:row*2]
        row = int(row/2)


haar = [[float(0) for _ in range(352)] for _ in range(352)]
haar = np.asarray(haar)

for i in range(0, 352):
    for j in range(0, 352, 8):
        haar[i, j:j+8] = HaarWaveletTransform(yplane[i, j:j+8].astype('float'))

haarfinal = [[float(0) for _ in range(352)] for _ in range(352)]
haarfinal = np.asarray(haarfinal)

for i in range(0, 352):
    for j in range(0, 352, 8):
        haarfinal[j:j+8, i] = HaarWaveletTransform(haar[j:j+8, i].astype('float'))

plt.imshow(haarfinal)

"""
compression ratio
"""
zerocountbefore = 0

for i in range(0,352):
    for j in range(0,352):
        if haarfinal[i][j] == 0 :
            zerocountbefore = zerocountbefore + 1
            
"""
threshold
"""
zerocountafter = 0

for i in range(0,352):
    for j in range(0,352):
        if abs(haarfinal[i][j]) < 10 :
            haarfinal[i][j] = 0
            zerocountafter = zerocountafter + 1
            
compressionRatio = float(123904-zerocountbefore)/float(123904 - zerocountafter)

"""
Inverse haar Trnasform of a row
"""
def InverseHaarWaveletTransform(x):
    row = 1
    matrix = [0.0]*8

    while True:
        for i in range(0,row):
            sum_val = (x[i] + x[row + i])
            diff = (x[i] - x[row+i])
            matrix[2*i] = sum_val
            matrix[2*i+1] = diff

        if row == 4:
            return matrix

        x[:row*2] = matrix[:row*2]
        row = int(row*2)


"""
Compression Value Based on some threshold Value
"""

#for i in range(0,344,8):
 #   for j in range(0,344,8):
  #      tempmatrix = haarfinal[i: i+8, j: j+8]
   #     mean = tempmatrix.mean() # mean
    #    std = tempmatrix.std() #standard deviation
     #   for x in (i,i+8):
      #      for y in (j,j+8):
       #         if abs(haarfinal[x][y]-mean) < std :
        #            haarfinal[x][y] = 0


haarinverse = [[float(0) for _ in range(352)] for _ in range(352)]
haarinverse = np.asarray(haarinverse)

for i in range(0, 352):
    for j in range(0, 352, 8):
        haarinverse[j:j+8, i] = InverseHaarWaveletTransform(haarfinal[j:j+8, i].astype('float'))

haarfinalinverse = [[float(0) for _ in range(352)] for _ in range(352)]
haarfinalinverse = np.asarray(haarfinalinverse)

for i in range(0, 352):
    for j in range(0, 352, 8):
        haarfinalinverse[i, j:j+8] = InverseHaarWaveletTransform(haarinverse[i, j:j+8].astype('float'))

plt.imshow(haarfinalinverse)
