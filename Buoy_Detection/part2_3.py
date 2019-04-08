# -*- coding: utf-8 -*-

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
from gaussian import Gaussian


def main():
    """
    Entry point of app
    """
    cap = cv2.VideoCapture("detectbuoy.avi")
    gaussian = Gaussian()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output_part_2_3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
    did_run_once = False
    i=0
    while cap.isOpened():
        i = i + 1
        ret, frame = cap.read()
        if frame is not None and not did_run_once:
            did_run_once = False
            frame = gaussian.detect_buoys(frame, i)
            cv2.imshow('Buoy Detection', frame)
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
