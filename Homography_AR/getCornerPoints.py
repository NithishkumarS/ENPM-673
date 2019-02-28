#!/usr/bin/env python3

__author__ = "Nantha Kumar Sunder, Nithish Kumar"
__version__ = "0.1.0"
__license__ = "MIT"

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial import distance as dist
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def getCornerPoints(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 240, 255, 0, cv2.THRESH_BINARY)
    try:
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dst_total = np.zeros(gray.shape, dtype='uint8')
    hierarchy = hierarchy[0]
    corner_points = []
    for j,cnt in zip(hierarchy,contours):
        currentContour = cnt
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            if j[3] != -1:

                gray_fl = np.float32(gray)
                mask = np.zeros(gray_fl.shape, dtype='uint8')
                mask = cv2.GaussianBlur(mask, (3, 3), 0)
                cv2.drawContours(frame, [currentContour], -1, (0, 255, 0), 3)
                cv2.fillPoly(mask, [currentContour], (255, 255, 255))
                dst = cv2.cornerHarris(mask, 5, 3, 0.04)
                dst = cv2.dilate(dst, None)
                ret, dst = cv2.threshold(dst, 0.1*dst.max(), 255, 0)
                dst_total = dst + dst_total
                corner_points.append(cnt)
    if corner_points:
        return corner_points[0], dst_total, frame
    else:
        return corner_points, dst_total, frame
