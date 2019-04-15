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
    cv2.imshow('frame after',frame)  
    input_frame = frame[cornerPoints[0][1]:cornerPoints[2][1],cornerPoints[0][0]:cornerPoints[2][0] ]
    template = tmpImg[cornerPoints[0][1]:cornerPoints[2][1],cornerPoints[0][0]:cornerPoints[2][0] ]
#     cv2.imshow('template',template)
#     cv2.imshow('innp_frame',input_frame)
    diff = template - input_frame

    #gradient X and Y
    gradX = np.uint8(cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=3))
    gradY = np.uint8(cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=3))
    cv2.imshow('gradX',gradX)
    cv2.imshow('gradY',gradY)
    

    
    print(input_frame.shape)
    op1 = np.zeros_like(input_frame)
    op = np.zeros(input_frame.shape[0]*input_frame.shape[1]*6)
#     op = np.zeros((input_frame.shape[0],input_frame.shape[1]))*6
    op = op.reshape((6,input_frame.shape[0], input_frame.shape[1]))
    print('op size:',op.shape)
    
    # Steepest Descent 
    for x in range(0,input_frame.shape[1]):
        for y in range(0,input_frame.shape[0]):
            tmp =  np.matmul(np.array([ [gradX[y][x],gradY[y][x]] ]),computeJacobian(x, y))#.astype(np.float32)
            
            for i in range(6):
                op[i][y][x]=tmp[0][i]

#                 op[y][x+(i*input_frame.shape[1])-1]=tmp[0][i]
    output_jacob = np.uint8(op.reshape((input_frame.shape[0]*6, input_frame.shape[1])))
    cv2.imshow('op',output_jacob)
    
    # Compute Hessian
    Hessian = np.zeros((input_frame.shape[1],input_frame.shape[1])).astype(np.float64)
    SD_param = np.zeros((input_frame.shape[1],input_frame.shape[1])).astype(np.float64)
    print('Hessian',Hessian.shape)
    print('OP:::::::::::::::::::',op.shape)
    
    for i in range(6):
        inp = np.matmul(op[i].T,op[i])
        Hessian += inp
    Hessian1 = [[np.sum(np.matmul(op[i].T,op[i])) for i in range(6)] for i in range(6)]  
#     Hessian = [[np.sum(np.multiply(op[a].T, op[b])) for a in range(6)] for b in range(6)]
    print(Hessian.shape)
    print(len(Hessian1))
    print('0',len(Hessian1[0]))
    
#     np.matmul(op,np.transpose(op))          # (nx6m) x (6mxn)  =nxn 
    print('Hes',Hessian1)
#     cv2.imshow('Hessian',Hessian)
    Hessian_Inv = np.linalg.pinv(Hessian1)
    
    cv2.imshow('HessianInv',Hessian_Inv)
    
    SD_param = np.zeros(6)
    for i in range(6):
        inp = np.matmul(op[i].T,diff)
        SD_param[i] = np.sum(inp)
    print(SD_param)
    
    delp = np.matmul(Hessian_Inv,SD_param)
    print('delp',delp)
    delp = delp.reshape((2,3))
    Warp = prevWarp + delp
    
    print('corner Ponts before:',cornerPoints)
    cornerPoints = [np.matmul(Warp,[x,y,1]).astype(int) for x,y in cornerPoints]
    print('corner Ponts after:',cornerPoints)

    
    
    '''
    for i in range(6):
        op[y][x+(i*input_frame.shape[1])-1]

    '''
    error = np.linalg.norm(delp)
    print('error: ',error)
    return cornerPoints, Warp, error