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
from homography import homographicTransform
from ARTag_Decoder import decode
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

def main():
    """ Main entry point of the app """
    cap = cv2.VideoCapture('Tag1.mp4')
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
        corner_idx_temp = np.zeros((4,1))
        corner_idx = np.zeros((1,1))
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
                print(corners)
                if (len(corners) < 4):
                    break
                #corner_idx_temp = sortCorners(corners)
                #corners = np.delete(corners, np.setdiff1d(np.arange(len(corners)), corner_idx_temp), axis = 0)
                corners = sortCorners(corners)
                corner_points = np.concatenate((corner_points, corners), axis = 0)
                #corner_idx = np.concatenate((corner_idx, corner_idx_temp), axis = 0)

        #corner_idx = np.delete(corner_idx, (0), axis=0)
        corner_points = np.delete(corner_points, (0), axis=0)
        #corner_idx = np.rint(corner_idx)
        #corner_idx = corner_idx.astype(int)
        corner_points = (np.rint(corner_points)).astype(int)
        #print('corner')
        #print(corner_idx)
        print('corner_points')
        print(corner_points)
        total_tags = np.int(len(corner_points)/4);
        print('total_tags')
        print(total_tags)
        for tag_no in range(0,total_tags):
            #print('inside each tag')
            #print((corner_points[4*tag_no:4*tag_no+4][:]) )
            #print((corner_idx[4*tag_no:4*tag_no+4]))

            H = homographicTransform(corner_points[4*tag_no:4*tag_no+4][:])#,(corner_idx[4*tag_no:4*tag_no+4]))
            transformed_image = np.zeros((200,200), dtype='uint8')
            h_inv = np.linalg.inv(H)
            for row in  range(0,200):
                for col in range(0,200):
                    X_dash = np.array([col,row,1]).T
                    X = np.matmul(h_inv,X_dash)
                    X = (X/X[2])
                    X = X.astype(int)
                    transformed_image[col][row] = gray[X[1]][X[0]]
            #cv2.imshow('QR_image',transformed_image)
            ID_val = decode(transformed_image)
            print(ID_val)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame_modi = frame
        frame_modi[dst_total>0.01*dst_total.max()]=[0,0,255]
        cv2.imshow('Harris corner detector', frame_modi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()

