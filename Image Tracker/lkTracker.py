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

def affineLKtracker(frame, template, cornerPoints, prevWarp,no):
    
    frame = cv2.warpAffine(frame, prevWarp, (frame.shape[1], frame.shape[0]))
    cv2.imshow('frame after',frame)  
    print('type:', type(cornerPoints))
    plotPoints = [i.astype(int) for i in cornerPoints]
    print('type:', type(plotPoints))
    input_frame = frame[plotPoints[0][1]:plotPoints[2][1],plotPoints[0][0]:plotPoints[2][0] ]
    template = tmpImg[int(cornerPoints[0][1]):int(cornerPoints[2][1]),int(cornerPoints[0][0]):int(cornerPoints[2][0]) ]

    cv2.imshow('inp img',input_frame)
    
    print(input_frame.shape)
    cv2.imshow('innp_frame',input_frame)
    print(input_frame.shape)
    input_frame = cv2.resize(input_frame,(template.shape[1],template.shape[0]))
    diff = template - input_frame
    print('input frame', input_frame.shape)
   
    #gradient X and Y
    gradX = cv2.Sobel(input_frame, cv2.CV_64F, 1, 0, ksize=3)
    gradY = cv2.Sobel(input_frame, cv2.CV_64F, 0, 1, ksize=3)
    cv2.imshow('gradX',gradX)
    cv2.imshow('gradY',gradY)
 
    print(input_frame.shape)
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
    output_jacob = op.reshape((input_frame.shape[0]*6, input_frame.shape[1]))
    cv2.imshow('op',np.uint8(output_jacob.T))
    cv2.waitKey(0)
    # Compute Hessian
    Hessian = np.zeros((input_frame.shape[1],input_frame.shape[1])) #.astype(np.float64)
    SD_param = np.zeros((input_frame.shape[1],input_frame.shape[1]))#.astype(np.float64)
    print('Hessian',Hessian.shape)
    print('OP:::::::::::::::::::',op.shape)
    '''
    for i in range(6):
        inp = np.matmul(op[i].T,op[i])
        Hessian += inp
    '''
    Hessian1 = [[np.sum(np.matmul(op[i].T,op[j])) for j in range(6)] for i in range(6)]  
    mag = np.linalg.norm(Hessian1)
   
    Hessian2 = (Hessian1/mag)*256.0
#     print('Hes: ',Hessian1)
    
#     cv2.imshow('Hessian',Hessian2)
#     np.matmul(op,np.transpose(op))          # (nx6m) x (6mxn)  =nxn 
    Hessian_Inv = np.linalg.pinv(Hessian1)
    mag2 = np.linalg.norm(Hessian_Inv)
    Hessian_Inv = (Hessian_Inv/mag2)*256.0
    
#     cv2.imshow('HessianInv',Hessian_Inv)
    SD = [np.sum(np.matmul(op[i].T,diff)) for i in range(6)]
    mag3 = np.linalg.norm(SD)
    '''
    SD_param = np.zeros(6)
    for i in range(6):
        inp = np.matmul(op[i].T,diff)
        SD_param[i] = np.sum(inp)
    print('SD_param',SD_param)
    print('SD:  ', SD)
    '''
    SD = np.array(SD/mag3)
    delp = np.matmul(Hessian_Inv,SD)
    print('delp',delp)
    delp = delp.reshape((2,3))
    Warp = prevWarp + delp
    '''
    print('corner Ponts before:',cornerPoints)
    cornerPoints = [np.matmul(Warp,[x,y,1]) for x,y in cornerPoints]
    print('corner Ponts after:',cornerPoints)
    '''
    '''
    for i in range(6):
        op[y][x+(i*input_frame.shape[1])-1]

    '''
    error = np.linalg.norm(delp)
    '''
    plt.plot(no,error,'xc')
    plt.draw()
    plt.pause(0.1)
    
    print('Warp:  ',Warp)
    '''
    print('error: ',error)
    return cornerPoints, Warp, error