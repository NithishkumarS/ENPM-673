#!/usr/bin/env python3

__author__ = "Nantha Kumar Sunder, Nithish Kumar"
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
from homography import homographicTransform
from homography import getTransfomredImage
from undistortion import get_undistort
from colorSegmentation import colorSegmentation
from houghTransform import houghTransform
def getVideoFile(usr_input):
    switcher = {
        1: 'challenge_video.mp4',
        2: 'project_video.mp4',
    }
    return switcher.get(usr_input, 'challenge_video.mp4')

def main():
    """ Main entry point of the app """
    usr_input = input(
        'Select the Video\n\t1. challenge_video.mp4 \n\t2. project_video.mp4 \n\nYour Choice: ')
    print(getVideoFile(int(usr_input)))
    cap = cv2.VideoCapture(getVideoFile(int(usr_input)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    Xc = np.array([[149, 0], [249, 0], [249, 399], [149, 399]])
    Xw = np.array([[548, 518], [761, 522], [891, 616], [408, 616]])
    kernel = np.ones((4,4),np.uint8)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            frame = np.array(frame, dtype=np.uint8)
            segmented_image = colorSegmentation(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            image_shape = gray.shape
            cropped_image = segmented_image.copy()
            cropped_image[0:int(image_shape[0]*2/3),:] = 1
            erosion = cv2.erode(segmented_image,kernel,iterations = 1)
            '''
            ret, thresh = cv2.threshold(cropped_image, 150, 255, 0, cv2.THRESH_BINARY)
            try:
                _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            '''
            '''
            Homography = homographicTransform(Xw, Xc)
            transformed_image = getTransfomredImage(np.linalg.inv(Homography[0]), gray, 400)
            undistorted_img = get_undistort(transformed_image)
            '''
            houghTransform(erosion, frame)    
            edges = cv2.Canny(cropped_image,100,200)
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
            cv2.imshow('transformed_image', erosion)
            cv2.imshow('Lane Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
