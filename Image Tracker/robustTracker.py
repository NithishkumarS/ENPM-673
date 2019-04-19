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


def scaleToAverageBrightness(frame, tmpImg, cornerPoints):
    """
    Scale the brightness of pixels in each frame so that the average
    brightness of pixels in the tracked region stays the same as
    the average brightness of pixels in the template.
    """
    avgB = np.sum(tmpImg)
    avgB = avgB/(tmpImg.shape[0] * tmpImg.shape[1])
    input_frame = frame[int(cornerPoints[0][1]):int(cornerPoints[2][1]),int(cornerPoints[0][0]):int(cornerPoints[2][0])]
    avgBI = np.sum(input_frame)
    avgBI = avgBI/(input_frame.shape[0] * input_frame.shape[1])
    scaling_factor = avgB/avgBI
    input_frame = input_frame * scaling_factor
    frame[int(cornerPoints[0][1]):int(cornerPoints[2][1]),int(cornerPoints[0][0]):int(cornerPoints[2][0])] = input_frame
    return frame


def huberLoss():
    """
    Huber loss function for outliers to minimize adverse affect
    on the cost function evaluation
    """
    pass
