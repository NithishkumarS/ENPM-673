#!/usr/bin/env python3
from openpyxl.drawing.image import bounding_box
__author__ = "Nantha Kumar Sunder, Nithish Kumar"
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
from boundingBox import *
from MSER import *

def last_4chars(x):
    return(x[-5:])

def loadImages():
    imageList = []
    for file in os.listdir("TSR/ProcessedInput/"):
        filename, basename = os.path.splitext(file)
        imageList.append(filename)
    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = "TSR/ProcessedInput/" + str(imageList[i]) + ".jpg"
    return imageList

def denoise(new_img):
    dst = cv2.fastNlMeansDenoisingColored(new_img,None,10,10,7,21)
    return dst

def getInp(new_img):
    mask_blue, mask_red = colorSegmentation(new_img)
    # kernel = np.ones((3,3),np.uint8)
    # erosion = cv2.erode(mask_blue,kernel,iterations = 1)
    # erosion = cv2.Canny(erosion,100,200)
    # blackhat = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    imb = cv2.cvtColor(mask_blue, cv2.COLOR_BGR2GRAY)

    imr = cv2.cvtColor(mask_red, cv2.COLOR_BGR2GRAY)
    im = imb | imr
    return im 

def main():
    """ Main entry point of the app """
    frameCount = 880
    imageList = loadImages()
    while frameCount < len(imageList):
        new_img = cv2.imread(imageList[frameCount])
        
        r_channel = new_img[:,:,2]
        g_channel = new_img[:,:,1]
        b_channel = new_img[:,:,0]
        
        sum = r_channel + g_channel + b_channel
        sum[sum==0] = 1
        r_channel = np.divide(r_channel, sum)
        g_channel = np.divide(b_channel, sum)
        b_channel = np.divide(b_channel, sum)
        
        r = np.uint8(np.maximum(0,np.minimum((r_channel-b_channel),(r_channel-g_channel)) ) )
        b = np.uint8(np.maximum(0,np.minimum((b_channel-r_channel),(b_channel-g_channel)) ) )
        
        ret, rr = cv2.threshold(r, 240, 255, cv2.THRESH_BINARY)
        ret, bb = cv2.threshold(b, 240, 255, cv2.THRESH_BINARY)
        rb = r | b

        cv2.imshow('rb',r)
        
        cv2.imshow('frame', new_img)
#         cv2.imshow('r',r)
#         cv2.waitKey(0)
         
#         r = np.copy(new_img)        #  r - 35-45    g- 60-70   b-100 - 115     
#         r[:,:,1] = 0
#         r[:,:,2] = 0
#         ret,th1 = cv2.threshold(r[:,:,0],100,115,cv2.THRESH_BINARY)
#         cv2.imshow('red_channel',th1)
#         inp = boundingBox(new_img)
        
#         inp = getInp(new_img)    
        frame = MSER(b, new_img)
#         boundingBox(new_img)
        cv2.imshow('frame', frame)
#         cv2.imshow('mask', inp)
        # cv2.imshow('segmented', res)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        frameCount = frameCount + 1
        print(frameCount)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
