#!/usr/bin/env python3

__author__ = "Nantha Kumar Sunder, Nithish Kumar"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import matplotlib.pyplot as plt
import os, sys
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def sortCorners(__corners__):
    corner_points = np.zeros((__corners__.shape))
    corner_points[0,:] = __corners__[np.argmin(__corners__[:,0], axis = 0),:]
    __corners__ = np.delete(__corners__, (np.argmin(__corners__[:,0])), axis = 0)
    corner_points[1,:] = __corners__[np.argmin(__corners__[:,1], axis = 0),:]
    __corners__ = np.delete(__corners__, (np.argmin(__corners__[:,1])), axis = 0)
    corner_points[2,:] = __corners__[np.argmax(__corners__[:,0], axis = 0),:]
    __corners__ = np.delete(__corners__, (np.argmax(__corners__[:,0])), axis = 0)
    corner_points[3,:] = __corners__[np.argmax(__corners__[:,1], axis = 0),:]
    __corners__ = np.delete(__corners__, (np.argmax(__corners__[:,1])), axis = 0)
    return corner_points

def getCornerPoints(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5), 0)
    ret, thresh = cv2.threshold(gray, 240, 255,0, cv2.THRESH_BINARY)
    _,contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    dst_total = np.zeros(gray.shape, dtype ='uint8')
    hierarchy = hierarchy[0]
    corner_points = np.zeros((1,2))
    for component in zip(contours, hierarchy):
        currentHierarchy = component[1]
        currentContour = component[0]
        size = cv2.minAreaRect(component[0])
        isSecond =0;
        idx = currentHierarchy[3]
        while True:
            if idx == -1:
                break
            isSecond = isSecond + 1
            idx = hierarchy[idx][3]
        if isSecond==1:
            gray_fl = np.float32(gray)
            mask = np.zeros(gray_fl.shape, dtype ='uint8')
            mask = cv2.GaussianBlur(mask,(3,3), 0)
            cv2.fillPoly(mask, [currentContour], (255,255,255))
            dst = cv2.cornerHarris(mask,5,3,0.04)
            dst = cv2.dilate(dst,None)
            ret, dst = cv2.threshold(dst, 0.1*dst.max(),255,0)
            dst = np.uint8(dst)
            cv2.drawContours(frame, [currentContour], -1, (0,255,0), 3)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(gray_fl, np.float32(centroids), (5,5), (-1,-1), criteria)
            dst_total = dst + dst_total
            corners = np.delete(corners, (0), axis=0)
            if (len(corners) < 4):
                break
            corners = sortCorners(corners)
            corner_points = np.concatenate((corner_points, corners), axis = 0)

    corner_points = np.delete(corner_points, (0), axis=0)
    corner_points = (np.rint(corner_points)).astype(int)
    return corner_points, dst_total, frame

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
