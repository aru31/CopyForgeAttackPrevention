#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:19:42 2018

@author: palakgoenka
"""
import os
import cv2
import numpy as np
import pandas as pd
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL.Image import Image
import matplotlib.image as mpimg
from skimage import io, transform
from skimage.io import imread
from skimage import data
from skimage.color import rgb2gray
from skimage import feature
from skimage import color
from scipy import ndimage as ndi
from skimage.segmentation import slic
from skimage.util import view_as_blocks
import math

"""
preprecessing step
"""
rgb_image = io.imread('test.jpg')
#rgb->lab
lab_image = color.rgb2lab(rgb_image)

# num_superpixel = k
num_superpixel = 1000
k = num_superpixel
# image size = x*y = n
image_size = rgb_image.shape[0]*rgb_image.shape[1]
n = image_size

# space between center of two superpixel
S = math.floor((( n ** 0.5 ) / (k ** 0.5))+0.5)

#Ck 5D matrix containg lab and super pixel coordinate
Ck = []
temp = []

x = S;
y = S;

temp.append(lab_image[x][y][0])
temp.append(lab_image[x][y][1])
temp.append(lab_image[x][y][2])
temp.append(x)
temp.append(y)
Ck.append(temp[0:])


for i in range(0,k):
    temp = []
    xi = x
    yi = y
    if x+S < rgb_image.shape[0] :
        xi = x+S
    if y+S < rgb_image.shape[1] :
        yi = y+S
    if xi!=x and yi!=y :
        temp.append(lab_image[xi][yi][0])
        temp.append(lab_image[xi][yi][1])
        temp.append(lab_image[xi][yi][2])
        temp.append(xi)
        temp.append(yi)
        Ck.append(temp[0:])
    if xi+S > rgb_image.shape[0] and yi+S > rgb_image.shape[1] :
        break
    x = xi
    y = yi
    
















