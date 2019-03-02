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

from ARTag_Decoder import decode
from getCornerPoints import getCornerPoints
from homography import homographicTransform
from part_1 import getTransfomredImage, getVideoFile
from virtualCube import virtualCube



def main():
    """ Main entry point of the app """
    usr_input = input(
        'Select the Video\n\t1. Tag0 \n\t2. Tag1 \n\t3. Tag2 \n\t4. multipleTags\n\nYour Choice: ')
    print(getVideoFile(int(usr_input)))
    cap = cv2.VideoCapture(getVideoFile(int(usr_input)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    Xc = np.array([[0, 0], [199, 0], [199, 199], [0, 199]])
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:

            frame = np.array(frame, dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            corner_points_temp, dst_total, frame = getCornerPoints(frame)
            for tag_no in range(0, (len(corner_points_temp))):
                corner_points = corner_points_temp[0]
                H = homographicTransform(corner_points, Xc)
                virtualCube(H, frame, corner_points[4*tag_no:4*tag_no+4][:])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
