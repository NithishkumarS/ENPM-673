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
    # Draw first 10 matches.
    # img3 = cv2.drawMatches(new_img,kp_new,old_img,kp_old,matches[:40],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches', img3)
    # cv2.waitKey(0)
    pts_new, pts_old = getMatchPoints(kp_old, kp_new, matches)
    return pts_new, pts_old


def getMatchPoints(kp_old, kp_new, matches):
    pts_new, pts_old = list(),list()
    for i in matches:
        pts_new.append(kp_new[i.trainIdx].pt)
        pts_old.append(kp_old[i.queryIdx].pt)
    return pts_new, pts_old
