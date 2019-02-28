#!/usr/bin/env python3

__author__ = "Nantha Kumar Sunder, Nithish Kumar, Rama Prashanth"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ARTag_Decoder import decode
from getCornerPoints import getCornerPoints
from homography import homographicTransform
from part_1 import getTransfomredImage, getVideoFile
from superimpose import superImpose

# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass


def main():
    """ Main entry point of the app """
    usr_input = input(
        'Select the Video\n\t1. Tag0 \n\t2. Tag1 \n\t3. Tag2 \n\t4. multipleTags\n\nYour Choice: ')
    print(getVideoFile(int(usr_input)))
    cap = cv2.VideoCapture(getVideoFile(int(usr_input)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    Xc = np.array([[0, 0], [511, 0], [511, 511], [0, 511]])
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imshow('Normal', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        corner_points, dst_total, frame = getCornerPoints(frame)
        for tag_no in range(0, np.int(len(corner_points)/4)):
            H = homographicTransform(corner_points[4*tag_no:4*tag_no+4][:], Xc)
            h_inv = np.linalg.inv(H)
            transformed_image = getTransfomredImage(h_inv, gray, 512)
            ID_val, rotations = decode(transformed_image)
            # print(['Tag ' + str(tag_no + 1 ) + ' value: ' + str(ID_val)])
            temp = frame
            # print('rotation:',rotations)
            frame = superImpose(h_inv, temp, rotations+1)
        # frame[dst_total>0.01*dst_total.max()]=[0,0,255]
        #cv2.imshow('Harris corner detector', frame)
        cv2.imshow('Superimposed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
