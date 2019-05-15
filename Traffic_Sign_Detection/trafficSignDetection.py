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
def denoise(new_img):
    dst = cv2.fastNlMeansDenoisingColored(new_img,None,10,10,7,21)
    return dst

def main():
    """ Main entry point of the app """
    frameCount = 0
    imageList = loadImages()
    while frameCount < len(imageList):
        new_img = cv2.imread(imageList[frameCount])
        cv2.imshow('frame', new_img)
        new_img = denoise(new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frameCount = frameCount + 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
