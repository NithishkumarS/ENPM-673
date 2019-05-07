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
import matplotlib.pyplot as plt
import numpy as np

def sift(new_img, old_img):
    new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
    old_img = cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp_new, des_new = sift.detectAndCompute(new_img,None)
    kp_old, des_old = sift.detectAndCompute(old_img, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_new,des_old,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
         if m.distance < 0.75*n.distance:
             good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(new_img,kp_new,old_img,kp_old,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches', img3)
    # cv2.waitKey(0)
    pts_new, pts_old = getMatchPoints(kp_old, kp_new, good)
    return 0


def orb(new_img, old_img):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp_new, des_new = orb.detectAndCompute(new_img,None)
    kp_old, des_old = orb.detectAndCompute(old_img,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des_new,des_old)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    matches = matches[:50]   
    '''
    img3 = cv2.drawMatches(new_img,kp_new,old_img,kp_old,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', img3)
    cv2.waitKey(0)
    '''
    pts_new, pts_old = getMatchPoints(kp_new, kp_old, matches)
    return pts_new, pts_old


def getMatchPoints(kp_new, kp_old, matches):
    pts_new, pts_old = list(),list()
    list_kp1, list_kp2 = list(),list()
    for i in matches:
        img1_idx = i.queryIdx
        img2_idx = i.trainIdx
        # x - columns
        # y - rows
        # Get the coordinates
        (x1, y1) = kp_new[img1_idx].pt
        (x2, y2) = kp_old[img2_idx].pt

        # Append to each list
        pts_new.append((x1, y1))
        pts_old.append((x2, y2))
        '''
        if i.trainIdx < len(kp_new) and i.queryIdx < len(kp_old):
            # pts_new.append(kp_new[i.trainIdx].pt)
            # pts_old.append(kp_old[i.queryIdx].pt)
            #
            pts_new.append([kp_new[i.trainIdx].pt[0], kp_new[i.trainIdx].pt[1]])
            pts_old.append([kp_old[i.queryIdx].pt[0], kp_old[i.queryIdx].pt[1]])
    print('len',len(pts_new))
    dd
    pts_new = np.array(pts_new)
    pts_old = np.array(pts_old)

    return pts_new, pts_old

    '''
    pts_new = np.array(pts_new)
    pts_old = np.array(pts_old)

    return pts_new, pts_old
