#!/usr/bin/env python3
__author__ = "Nantha Kumar Sunder, Nithish Kumar"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def last_4chars(x):
    return(x[-5:])

def loadImages():
    imageList = []
    for file in os.listdir("TSR/input/"):
        filename, basename = os.path.splitext(file)
        imageList.append(filename)
    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = "TSR/input/" + str(imageList[i]) + ".jpg"
    return imageList

def denoise():
    path = os.path.dirname(os.path.realpath(__file__))
    imageList = loadImages()
    frameCount = 0
    while frameCount < len(imageList):
        new_img = cv2.imread(imageList[frameCount])
        new_img = cv2.fastNlMeansDenoisingColored(new_img,None,10,10,7,15)
        cv2.imwrite('TSR/ProcessedInput/image.0'+str(32640+frameCount)+'.jpg', new_img)
        print(frameCount)
        frameCount += 1

def main():
    denoise()

if __name__ == "__main__":
    main()
