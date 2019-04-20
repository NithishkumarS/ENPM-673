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
from lkTracker import afflineInv
from robustTracker import scaleToAverageBrightness

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

    num = 0
    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = videos[option] + imageList[i] + basename
    if option == '1':
        cornerPoints = np.array([[127,105],[127,274],[332,274],[332,105]])
        num = 1
    elif option == '2':
        cornerPoints = np.array([[258,295],[285,295],[285,360],[258,360]])
        num = 2
    else:
        cornerPoints = np.array([[120,68],[120,156],[178,156],[178,68]])
        num = 2
    tmpImg = cv2.imread(imageList[0])

    return imageList, cornerPoints, tmpImg, num

def main():
    """ Main entry point of the app """
    imageList = input('Select the Video\n 1. Car \n 2. Person \n 3. Vase \n=>')
    imageList, cornerPoints, tmpImg, num = loadImages(imageList)
    totalFrame = len(imageList)
    frameCount = 1
    temp = cv2.imread(imageList[0])
    plotPoints = [i for i in cornerPoints]
    thresh = [.0001, .035, .00215]     # vase: .0215 .015:
    frame_width = temp.shape[0]
    frame_height = temp.shape[1]
    out = cv2.VideoWriter('output_part_3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
    
    while frameCount < len(imageList):
        prevWarp = np.array([[1,0,-0.25],[0,1,0.05]])
        tmpImg = cv2.imread(imageList[frameCount-1],cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(imageList[frameCount],cv2.IMREAD_GRAYSCALE)

       
        error = 1
        iter = 0
        no = 0
        while error > thresh[num-1]:          
            print('frame cout', frameCount)
            if frameCount != 0:
                no = no + 1
                prevWarp, error =  affineLKtracker(frame, tmpImg, plotPoints, prevWarp,no)
                print('no:', iter)
            iter += 1
            if error == -1:
#                 print('Singular hessian')
                break
            if iter > 600:
                break
        tempPoints  = np.array([np.matmul(afflineInv(prevWarp),[x,y,1]) for x,y in plotPoints])
        tempPoints = np.round(tempPoints.astype(int))
        plotPoints = [i.astype(int) for i in tempPoints]
        pts = tempPoints.copy()
        pts = pts.reshape((-1,1,2))
        colorImage = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
        cv2.polylines(colorImage,[pts],True,(0,0,255))
        plotPoints = [np.round(i).astype(int) for i in tempPoints] 
        plot2 = cv2.rectangle(frame,(plotPoints[0][0],plotPoints[0][1]),(plotPoints[2][0],plotPoints[2][1]),[0,0,255])
        out.write(colorImage)
        cv2.imshow('frame', colorImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('error',error)
        frameCount = frameCount + 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
