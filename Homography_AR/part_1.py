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

def getVideoFile(usr_input):
    switcher =  {
        1: 'Input Sequences/Tag0.mp4',
        2: 'Input Sequences/Tag1.mp4',
        3: 'Input Sequences/Tag2.mp4',
        4: 'Input Sequences/multipleTags.mp4',
    }
    return switcher.get(usr_input, 'Input Sequences/Tag0.mp4' )

def getTransfomredImage(h_inv, gray,n):
    transformed_image = np.zeros((n,n), dtype='uint8')
    for row in  range(0,n):
        for col in range(0,n):
            Xc = np.array([col,row,1]).T
            Xw = np.matmul(h_inv,Xc)
            Xw = (Xw/Xw[2])
            Xw = Xw.astype(int)
            transformed_image[row][col] = gray[Xw[1]][Xw[0]]
    #print(transformed_image)
    return transformed_image

def main():
    """ Main entry point of the app """
    usr_input = input('Select the Video\n\n\t1. Tag0 \n\t2. Tag1 \n\t3. Tag2 \n\t4. multipleTags')
    print(getVideoFile(int(usr_input)))
    cap = cv2.VideoCapture(getVideoFile(int(usr_input)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    Xc = np.array([[0, 0], [0, 199], [199, 199], [199, 0]])
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = np.array(frame, dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(5,5), 0)
        corner_points, dst_total, frame = getCornerPoints(frame)
        #print(corner_points.shape)
        for tag_no in range(0, np.int(len(corner_points)/4)):
            H = homographicTransform(corner_points[4*tag_no:4*tag_no+4][:], Xc)
            h_inv = np.linalg.inv(H)
            transformed_image = getTransfomredImage(h_inv, gray,200)
            ID_val,rotation = decode(transformed_image)
            cv2.imshow('transformed_image',transformed_image)
            cv2.putText(frame,'Tag ' + str(tag_no + 1) + ' value: ' + str(ID_val),(10,100 + 50*(2*tag_no-1)), font, 2, (200,255,155), 2, cv2.LINE_AA)
        frame[dst_total>0.01*dst_total.max()]=[0,0,255]
        cv2.imshow('Harris corner detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
