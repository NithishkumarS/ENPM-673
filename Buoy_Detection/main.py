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

    did_run_once = False

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None and not did_run_once:
            did_run_once = True
            frame = gaussian.detect_buoys(frame)
            cv2.imshow('Buoy Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
