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
from homography import homographicTransform
from ARTag_Decoder import decode
from getCornerPoints import getCornerPoints
from part_1 import getTransfomredImage
from part_1 import getVideoFile
from virtualCube import virtualCube

def main():
    """ Main entry point of the app """
    usr_input = input('Select the Video\n\n\t1. Tag0 \n\t2. Tag1 \n\t3. Tag2 \n\t4. multipleTags')
    print(getVideoFile(int(usr_input)))
    cap = cv2.VideoCapture(getVideoFile(int(usr_input)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    Xc = np.array([[0, 0], [199, 0], [199, 199], [0, 199]])
    while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('Normal', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5), 0)
        corner_points, dst_total, frame = getCornerPoints(frame)
        for tag_no in range(0, np.int(len(corner_points)/4)):
            H = homographicTransform(corner_points[4*tag_no:4*tag_no+4][:],Xc)
            virtualCube(H,frame,corner_points[4*tag_no:4*tag_no+4][:])
        #cv2.imshow('Superimposed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
