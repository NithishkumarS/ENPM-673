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
import matplotlib.pyplot as plt

def computeJacobian(x,y):
    a = np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]])
    return a

def afflineInv(prevWarp):
    R = prevWarp[:,0:2]
    rinv = np.linalg.inv(R)
    trans = np.matmul(rinv,prevWarp[:,2])
    pinv = np.array( [ [rinv[0,0], rinv[0,1] , -trans[0]],  [rinv[1,0], rinv[1,1] , -trans[1]]  ])
    return pinv


def affineLKtracker(frame, tmpImg, cornerPoints, prevWarp,no):
    
    frame = cv2.warpAffine(frame, prevWarp, (frame.shape[1], frame.shape[0]))
#     img = np.copy(frame)
#     ploti = cv2.rectangle(img,(cornerPoints[0][0],cornerPoints[0][1]),(cornerPoints[2][0],cornerPoints[2][1]),[0,0,255])     cv2.imshow('After Warp frame', ploti)
       
    cv2.imshow('After warp',frame)
    input_frame = frame[int(cornerPoints[0][1]):int(cornerPoints[2][1]),int(cornerPoints[0][0]):int(cornerPoints[2][0]) ]
    template = tmpImg[int(cornerPoints[0][1]):int(cornerPoints[2][1]),int(cornerPoints[0][0]):int(cornerPoints[2][0]) ]
    diff = template - input_frame
    if diff.shape[0] == 0 or diff.shape[1]==0:
        return prevWarp, -1
    cv2.imshow('diff',diff)
    
    #gradient X and Y
    gradX = cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=5)
    gradY = cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=5)


    Height, Weight = template.shape
    Xc = np.tile(np.linspace(0, Weight-1, Weight), (Height, 1)).flatten()
    Yc = np.tile(np.linspace(0, Height-1, Height), (Weight, 1)).T.flatten()

    # Step 5 - Compute the steepest descent images
    steepest_descent = np.vstack([gradX.ravel() * Xc, gradY.ravel() * Xc,
        gradX.ravel()*Yc, gradY.ravel()*Yc, gradX.ravel(), gradY.ravel()]).T
#     cv2.imshow('steepest descent',steepest_descent)

    # Step 6 - Compute the Hessian matrix
    hessian = np.matmul(steepest_descent.T, steepest_descent)
#     mag = np.linalg.norm(hessian)
# 
#     hessian2 = (hessian/mag)*256.0
#     cv2.imshow('Hessian',hessian2)

    det_hessian = np.linalg.det(hessian)
    if det_hessian == 0:
        return prevWarp, -1

    # Step 7/8 - Compute delta P
    delp = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, diff.flatten()))
    
    Warp = prevWarp + np.array([[delp[0],delp[2],delp[4]], [delp[1],delp[3],delp[5]]])
    print('delp: ',np.array([[delp[0],delp[2],delp[4]], [delp[1],delp[3],delp[5]]]))

    error = np.linalg.norm(delp)
  
    print('error: ',error)
    return Warp, error
