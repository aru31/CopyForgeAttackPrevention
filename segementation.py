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
num_clusters = 100
height = gray.shape[0]
width = gray.shape[1]
numpixels = height * width
s_interval = int(math.sqrt(numpixels/num_clusters))


"""
Initializing in Making Clusters as a list represented by a
cluster center
First covering the width of one row and then the other
Made a function because of their repeated Use
"""
clusters = []

def initializeCluster():
    h = int(s_interval/2)
    w = int(s_interval/2)
    while h<height:
        while w<width:
            clusters.append([h, w, lab[h][w][0], lab[h][w][1], lab[h][w][2]])
            print(clusters)
            w = w + s_interval
        w = int(s_interval/2)
        h = h + s_interval


"""
Made a function because of their repeated Use
Gradient According to the Paper
"""
def gradient(h, w):
    if w+1 >= width:
        w = width - 2
    if h+1 >= height:
        h = height - 2
    if h<=0:
        h = 0
    if w<=0:
        w = 0

    gradient = math.pow((lab[h+1][w][0]-lab[h-1][w][0]), 2) + math.pow((lab[h+1][w][1]-lab[h-1][w][1]), 2) + math.pow((lab[h+1][w][2]-lab[h-1][w][2]), 2) + \
               math.pow((lab[h][w+1][0]-lab[h][w-1][0]), 2) + math.pow((lab[h][w+1][1]-lab[h][w-1][1]), 2) + math.pow((lab[h][w+1][2]-lab[h][w-1][2]), 2)
    return gradient


"""
Made a function because of their repeated Use
Moving Clusters According to Paper to lowest gradient 
position in 3*3 neighbourhood
"""
def moveCluster():
    for cluster in clusters:
        grad = gradient(cluster[0], cluster[1])
        for moveh in range(-1, 2):
            for movew in range(-1, 2):
                newh = moveh + cluster[0]
                neww = movew + cluster[1]
                newgrad = gradient(newh, neww)
                
                if grad > newgrad:
                    cluster[0] = newh
                    cluster[1] = neww
                    cluster[2] = lab[newh][neww][0]
                    cluster[3] = lab[newh][neww][1]
                    cluster[4] = lab[newh][neww][2]
                    grad = newgrad

"""
Made a function because of their repeated Use
Euclidean Distance donot work, so calculated differntly
i.e asscociating every pixel of the image to it's center
for CIELAB
"""
dist = np.full((height, width), np.inf)
label = {}
m = 10
imgpixel = []

# Definitely a million $ Algorithm... nailed it
def distance():
    i=-1
    for cluster in clusters:
        imgpixel.append([])
        i = i+1
        for h in range(cluster[0] - (2 * s_interval), cluster[0] + (2 * s_interval)):
            if h<0 or h>=height:
                continue
            for w in range(cluster[1] - (2 * s_interval), cluster[1] + (2 * s_interval)):
                if w<0 or w>=width:
                    continue
                l = lab[h][w][0]
                a = lab[h][w][1]
                b = lab[h][w][2]

                dlab = math.sqrt(math.pow(l - cluster[2], 2) + math.pow(a - cluster[3], 2) + math.pow(b - cluster[4], 2))
                dxy = math.sqrt(math.pow(h - cluster[0], 2) + math.pow(w - cluster[1], 2))
                d = dlab + (m/s_interval)*dxy
                if d<dist[h][w]:
                    for j in range(0, i):
                        if (h, w) in imgpixel[j]:
                            imgpixel[j].remove((h, w))
                        else:
                            imgpixel[i].append((h, w))
                    dist[h][w] = d
            

"""
Made a function because of their repeated Use
Updating New Cluster Centres
"""
def newCluster():
    i=-1
    for cluster in clusters:
        new_h = 0
        new_w = 0
        num = 0
        i = i+1
        for pixel in imgpixel[i]:
            new_h = newh + pixel[0]
            new_w = new_w + pixel[1]
            num = num + 1
            n_h = new_h / num
            n_w = new_w / num
            cluster[0] = n_h
            cluster[1] = n_w
            cluster[2] = lab[n_h][n_w][0]
            cluster[3] = lab[n_h][n_w][1]
            cluster[4] = lab[n_h][n_w][2]


"""
Iterating the Number of times required to call the above function
Just implemented K-Means Algorithm WOW!
"""
def iterationCluster():
    iterations = 10
    initializeCluster()
    moveCluster()
    for i in range(iterations):
        distance()
        newCluster()

# Moment Of Truth
iterationCluster()

"""
New Image that we get after finally getting SuperPixels
"""
def finalImage():
    newimg = np.copy(img)
    i=-1
    for cluster in clusters:
        i = i+1
        for pixel in imgpixel[i]:
            newimg[pixel[0]][pixel[w]][0] = cluster[0]
            newimg[pixel[0]][pixel[w]][1] = cluster[1]
            newimg[pixel[0]][pixel[w]][2] = cluster[2]
        newimg[cluster[0]][cluster[1]][0] = 0
        newimg[cluster[0]][cluster[1]][1] = 0
        newimg[cluster[0]][clustep[2]][2] = 0
    
    newrgbimg = color.lab2rgb(newimg)
    return newrgbimg

"""
SLIC Implemented
"""
