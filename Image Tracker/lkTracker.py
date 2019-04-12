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

def computeJacobian(x,y):
    a = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
    return a

def affineLKtracker(frame, tmpImg, cornerPoints, prevWarp):
    frame = cv2.warpAffine(frame, prevWarp, (frame.shape[1], frame.shape[0]))
#     cv2.imshow('frame after',frame)  
    template = tmpImg[cornerPoints[0][1]:cornerPoints[2][1],cornerPoints[0][0]:cornerPoints[2][0] ]
    
    #gradient X and Y
    gradX = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=5)
    gradY = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=5)
    cv2.imshow('gradX',gradX)
    cv2.imshow('gradY',gradY)
    
    print(frame.shape)
    op1 = np.zeros_like(frame)
    op = np.zeros((frame.shape[0],frame.shape[1]*6))
    print(op.shape)
    
    # Steepest Descent 
    for x in range(0,frame.shape[1]):
        for y in range(0,frame.shape[0]):
            tmp =  np.matmul(np.array([ [gradX[y][x],gradY[y][x]] ]),computeJacobian(x, y))#.astype(np.float32)
            
            for i in range(6):
                op[y][x+(i*720)-1]=tmp[0][i]
#     cv2.imshow('op',op)
    
    # Compute Hessian
    
    Hessian = np.matmul(op,np.transpose(op))          # (nx6m) x (6mxn)  =nxn 
#     cv2.imshow('Hessian',Hessian)
    
    HessianInv = np.linalg.inv(Hessian)
    
    
    error = 1
    Warp = prevWarp
    return cornerPoints, Warp, error