#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 09 2018 11:01:00 

@author: aruroxx31 and palak
"""


import os
import glob
import shutil

path = os.getcwd()
regex = re.compile(r"(?<![-.])\b[0-9]+\b(?!\.[0-9])")

num = []

for i in range(1, 101):
    num.append(str(i))

list_of_non_manipulated = [] 

for i in range(0, 100):
    list_of_non_manipulated.append(glob.glob(path+'/'+num[i]+'.tif'))


list_of_manipulated = [] 

for i in range(0, 100):
    list_of_manipulated.append(glob.glob(path+'/'+num[i]+'t.tif'))
    
manipath = path+'/manipulated'
notmanipath = path+'/notmanipulated'

new_manilist = []
for file in list_of_manipulated:
    file = "".join(str(x) for x in file)
    new_manilist.append(file)

new_nonmanilist = []
for file in list_of_non_manipulated:
    file = "".join(str(x) for x in file)
    new_nonmanilist.append(file)


for file in new_nonmanilist:
    shutil.move((file), notmanipath)

for file in new_manilist:
    shutil.move((file), manipath)