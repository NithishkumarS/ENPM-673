#!/usr/bin/env python3
__author__ = "Nantha Kumar Sunder, Nithish Kumar, Rama Prashanth"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from sift import sift

def loadImages():
    imageList = []
    for file in os.listdir("Oxford_dataset/stereo/Color"):
        filename, basename = os.path.splitext(file)
        imageList.append(int(filename))
    imageList = sorted(imageList)
    for i in range(len(imageList)):
        imageList[i] = "Oxford_dataset/stereo/Color/" + str(imageList[i]) + ".png"
    return imageList

def main():
    """ Main entry point of the app """
    frameCount = 50
    imageList = loadImages()
    old_image = cv2.imread(imageList[49])
    while frameCount < 51: #len(imageList):
        new_img = cv2.imread(imageList[frameCount])
        pt_new = sift(new_img)
        pt_old = sift(old_image)
        cv2.imshow('frame', new_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        frameCount = frameCount + 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Video extraction
# frame_width = temp.shape[1]
# frame_height = temp.shape[0]
# out = cv2.VideoWriter('Visual_Odometry.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
# out.write(colorImage)