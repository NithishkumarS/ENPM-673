#!/usr/bin/env python3
#=====================================
# Author: Nantha Kumar Sunder
# Description:
#=====================================

__author__ = "Nantha Kumar Sunder"
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

def main():
    """ Main entry point of the app """
    cap = cv2.VideoCapture('multipleTags.mp4')
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5), 0)
        ret, thresh = cv2.threshold(gray, 240, 255,0, cv2.THRESH_BINARY)
        contours, heirarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        dst_total = np.zeros(gray.shape, dtype ='uint8')
        heirarchy = heirarchy[0]
        corner = np.zeros(4)
        corner_points = np.zeros((1,2))
        for component in zip(contours, heirarchy):

            currentHierarchy = component[1]
            currentContour = component[0]
            size = cv2.minAreaRect(component[0])
            isSecond =0;
            idx = currentHierarchy[3]
            while True:
                if idx == -1:
                    break
                isSecond = isSecond + 1
                idx = heirarchy[idx][3]
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
                temp_corners = np.arange(len(corners))
                corner[0] = np.argmax(corners[:,0], axis = 0)
                corner[1] = np.argmin(corners[:,0], axis = 0)
                corner[2] = np.argmax(corners[:,1], axis = 0)
                corner[3] = np.argmin(corners[:,1], axis = 0)
                corners = np.delete(corners, np.setdiff1d(corners, temp_corners), axis = 0)
                corner_points = np.concatenate((corner_points, corners), axis = 0)

        corner_points = np.delete(corner_points, (0), axis=0)
        frame_modi = frame
        frame_modi[dst_total>0.01*dst_total.max()]=[0,0,255]
        cv2.imshow('frame', dst)
        cv2.imshow('Harris corner detector', frame_modi)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
