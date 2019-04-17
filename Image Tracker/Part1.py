#!/usr/bin/env python3
# from hgext.mq import prev

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

import matplotlib.pyplot as plt
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
    print(option)
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
    frameCount = 1
    prevWarp = np.array([[1,0,0.1],[0,1,-0.1]])
    
    while frameCount < len(imageList):
        tmpImg = cv2.imread(imageList[frameCount-1],cv2.IMREAD_GRAYSCALE)
      #  tmpImg = cv2.imread('/home/nithish/pyEnv/eclipse_py/Perception/ENPM-673/Image Tracker/data/car/frame0020.jpg',cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(imageList[frameCount],cv2.IMREAD_GRAYSCALE)
        #frame = cv2.imread('/home/nithish/pyEnv/eclipse_py/Perception/ENPM-673/Image Tracker/data/car/frame0021.jpg',cv2.IMREAD_GRAYSCALE)
        error = 1
        c = 0
#         plt.ion()
        no = 0
        plotPoints = [i.astype(int) for i in cornerPoints]

        while c < 8:#error > .001:
            if frameCount != 0:
                no = no + 1 
                prevWarp, error = affineLKtracker(frame, tmpImg, plotPoints, prevWarp,no)
                print('no:', no)
            c += 1
        
        print('corner Ponts before:',cornerPoints)
        cornerPoints = [np.matmul(prevWarp,[x,y,1]) for x,y in cornerPoints]
        print('corner Ponts after:',cornerPoints)
        plotPoints = [i.astype(int) for i in cornerPoints]
#                  print('plot points:', plotPoints)
        ploti = cv2.rectangle(frame,(plotPoints[0][0],plotPoints[0][1]),(plotPoints[2][0],plotPoints[2][1]),[0,0,255])
        cv2.imshow('frame', ploti)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

        frameCount = frameCount + 1
    plt.show(block=True)    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()