#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 07 17:17:56 2018

@author: aruroxx31 and palak
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

img=mpimg.imread('test.jpg');
plt.imshow(img)

ycbcr = rgb2ycbcr(img)
ycbcr = ycbcr[0: 352, 0: 352]
yplaneimg = []
yplaneimg.append(ycbcr[0: , 0:, 0])
yplaneimg = np.asarray(yplaneimg)
yplaneimg = yplaneimg.reshape(352, 352)

plt.imshow(yplaneimg)

hartranformmatrix = [[float(0) for _ in range(yplaneimg.shape[0])] for _ in range(yplaneimg.shape[0])]
hartranformmatrix = np.asarray(hartranformmatrix)
for x in range(0,yplaneimg.shape[0],8):
    for c in range(0,8):
        for y in range(0,yplaneimg.shape[0]):
            hartranformmatrix[x+c][y] = yplaneimg[x+c][y]
        for y in range(0,yplaneimg.shape[0]-8,8):
            for z in range(0,8):
                if z < 4:
                    hartranformmatrix[x+c][y+z] = float(hartranformmatrix[x+c][y+2*z] + hartranformmatrix[x+c][y+2*z+1])/float(2)
                else:
                    hartranformmatrix[x+c][y+z] = hartranformmatrix[x+c][y+2*z] - hartranformmatrix[x+c][y+z-4]

