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


    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = videos[option] + imageList[i] + basename
    if option == '1':
#         cornerPoints = np.array([[122,101],[122,278],[341,278],[341,101]])
        cornerPoints = np.array([[127,105],[127,274],[332,274],[332,105]])
    elif option == '2':
        cornerPoints = np.array([[258,295],[285,295],[285,360],[258,360]])
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
    plotPoints = [i for i in cornerPoints]
       
    while frameCount < len(imageList):
#         prevWarp = np.array([[1,0,-0.01],[0,1,0.01]])
        prevWarp = np.array([[1,0,0.01],[0,1,0.01]])
        tmpImg = cv2.imread(imageList[frameCount-1],cv2.IMREAD_GRAYSCALE)
#         tmpImg = cv2.imread('/home/nithish/pyEnv/eclipse_py/Perception/ENPM-673/Image Tracker/data/human/0140.jpg',cv2.IMREAD_GRAYSCALE)
        frame = cv2.imread(imageList[frameCount],cv2.IMREAD_GRAYSCALE)
#         img1 = np.copy(tmpImg)
#         ploti = cv2.rectangle(img1,(cornerPoints[0][0],cornerPoints[0][1]),(cornerPoints[2][0],cornerPoints[2][1]),[0,0,255])
#         cv2.imshow('initial frame', ploti)
#         frame = cv2.imread('/home/nithish/pyEnv/eclipse_py/Perception/ENPM-673/Image Tracker/data/human/0141.jpg',cv2.IMREAD_GRAYSCALE)
#         frame = scaleToAverageBrightness(frame, tmpImg, plotPoints)
       
        error = 1
        iter = 0
#         plt.ion()
        no = 0
        
        while error > .0215:           # human .05:    #    .0001:          # vase: .0215 
            print('frame cout', frameCount)
            if frameCount != 0:
                no = no + 1
                prevWarp, error =  affineLKtracker(frame, tmpImg, plotPoints, prevWarp,no)
                print('no:', iter)
            iter += 1
            if error == -1:
                print('Singular hessian')
                break
            if iter > 1000:
                break
        tempPoints  = np.array([np.matmul(afflineInv(prevWarp),[x,y,1]) for x,y in plotPoints])
#         # print('corner Points before:',cornerPoints)
        tempPoints = np.round(tempPoints.astype(int))
        plotPoints = [i.astype(int) for i in tempPoints]
        pts = tempPoints.copy()
        pts = pts.reshape((-1,1,2))
        cv2.polylines(frame,[pts],True,(0))
        
        '''



        plotPoints = [np.round(i).astype(int) for i in tempPoints]        
#                   print('plot points:', plotPoints)
        plot2 = cv2.rectangle(frame,(plotPoints[0][0],plotPoints[0][1]),(plotPoints[2][0],plotPoints[2][1]),[0,0,255])
        '''
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
        print('error',error)
      
        frameCount = frameCount + 1
#     plt.show(block=True)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
