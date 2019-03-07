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

    Xw = np.array([[600, 452], [683, 452], [1005, 630], [387, 630]])
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            frame = np.array(frame, dtype=np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            image_shape = gray.shape
            '''
            ret, thresh = cv2.threshold(cropped_image, 150, 255, 0, cv2.THRESH_BINARY)
            try:
                _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            '''

            undistorted_img = get_undistort(frame)
            segmented_image = colorSegmentation(undistorted_img)

            cropped_image = segmented_image.copy()
            cropped_image[0:int(image_shape[0]*1/2),:] = 1
            erosion = cv2.erode(segmented_image,kernel,iterations = 1)
            houghTransform(erosion, frame)
            Homography = homographicTransform(Xw, Xc)
            transformed_image = getTransfomredImage(np.linalg.inv(Homography[0]), segmented_image, 400,400)
            hist = cv2.calcHist([transformed_image],[0],None,[2],[0,2])
            edges = cv2.Canny(cropped_image,100,200)
            #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
            cv2.imshow('transformed_image', transformed_image)
            plt.plot(hist)
            plt.show()
            cv2.imshow('Lane Detection', frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
