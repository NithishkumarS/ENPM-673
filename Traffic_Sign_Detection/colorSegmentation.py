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

    return total_mask
