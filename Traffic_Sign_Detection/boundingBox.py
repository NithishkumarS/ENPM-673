#!/usr/bin/env python3

import os
import sys
import numpy as np
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2

def colorSegmentation(image):
    '''
        function to color segment for red and blue color using hsv
        input: Denoised image
        output: mask image
    '''
    hsv = cv2.cvtColor(image,  cv2.COLOR_BGR2HSV).astype(np.float)

    #--------------------- Blue mask ---------------------------
    lower_blue = np.array([60,100,100])
    upper_blue = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    blue_mask = cv2.bitwise_and(image,image, mask= mask)
    #-----------------------------------------------------------

    # -------------------- Red mask ----------------------------
    # lower mask
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask_light_red = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask_dark_red = cv2.inRange(hsv, lower_red, upper_red)

    # combining mask
    red_mask = cv2.bitwise_and(image, image, mask = mask_light_red + mask_dark_red)
    #-----------------------------------------------------------

    # total mask
    total_mask = cv2.bitwise_or(blue_mask, red_mask)

    return blue_mask, red_mask

def boundingBox(image):
    mask_blue, mask_red = colorSegmentation(image)
    # kernel = np.ones((3,3),np.uint8)
    # erosion = cv2.erode(mask_blue,kernel,iterations = 1)
    # erosion = cv2.Canny(erosion,100,200)
    # blackhat = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    im = cv2.cvtColor(mask_blue, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 5, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h >= 0.9*w and w*h > 100 and (h < 2.5*w) and w*h < 30000:
            print('height, width:', h,w)
            print("area", w*h)
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

    im = cv2.cvtColor(mask_red, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 5, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h >= 0.9*w and w*h > 100 and (h < 2.5*w) and w*h < 30000:
            print('height, width:', h,w)
            print("area", w*h)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)


    # cv2.drawContours(mask_blue, contours, -1, (0,255,0), 3)
    cv2.imshow('mask_blue', image)
    # cv2.imshow('mask_red', mask_red)
    cv2.waitKey(0)
    return 0
