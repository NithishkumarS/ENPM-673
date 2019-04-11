#!/usr/bin/env python3

__author__ = "Nantha Kumar Sunder, Nithish Kumar, Rama Prashanth"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys

# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import numpy as np
from lkTracker import affineLKtracker

def last_4chars(x):
    return(x[-4:])

def loadImages(option):
    imageList = []
    videos = {
    '1': 'data/car/',
    '2': 'data/human/',
    '3': 'data/vase/',
    }
    for file in os.listdir(videos[option]):
        filename, basename = os.path.splitext(file)
        imageList.append(filename)


    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = videos[option] + imageList[i] + basename
    # print(imageList)
    if option == '1':
        cornerPoints = np.array([[122,101],[122,278],[341,278],[341,101]])
    elif option == '2':
        cornerPoints = np.array([[256,293],[256,364],[288,364],[256,293]])
    else:
        cornerPoints = np.array([[120,68],[120,156],[178,156],[178,68]])

    tmpImg = cv2.imread(imageList[0])

    return imageList, cornerPoints, tmpImg

def main():
    """ Main entry point of the app """
    imageList = input('Select the Video\n 1. Car \n 2. Person \n 3. Vase \n=>')
    imageList, cornerPoints, tmpImg = loadImages(imageList)
    totalFrame = len(imageList)
    frameCount = 0
    prevWarp = np.zeros((1,1))
    while frameCount < len(imageList):
        frame = cv2.imread(imageList[frameCount])
        if frameCount != 0:
            frame, prevWarp = affineLKtracker(frame, tmpImg, cornerPoints, prevWarp)
        else:
            prevWarp = cornerPoints
        cv2.imshow('frame', frame)
        frameCount = frameCount + 1
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
