#!/usr/bin/env python3

import os
import sys
import numpy as np
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
import mahotas


def detectTrafficSign(frame):
    '''
    Detects blobs with blue color on frame and
    deduces largest square blob as the sign.
    '''
    # frame = imutils.resize(frame, width=500)
    # frameArea = frame.shape[0]*frame.shape[1]

    # convert color image to HSV color scheme
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV Filter Red and Blue Traffic Signs
    mask_blue = cv2.inRange(hsv, (100, 120, 100), (120, 255, 255))

    mask_red_lower = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
    mask_red_upper = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))

    mask = cv2.add(mask_red_lower, mask_red_upper)
    mask = cv2.add(mask, mask_blue)

    # Apply Gausian blur
    mask = cv2.GaussianBlur(mask, (11, 11), 0) # 3,3
    # T = mahotas.thresholding.otsu(mask)

    # Canny
    # mask = cv2.Canny(mask, T * 0.5, T)
    minW = 15
    minH = 15

    # Find contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    #
    # # Draw contours
    # if len (cnts) > 0:
    #     mask = cv2.drawContours(mask, cnts, -1, 255, -1)
    #     # Erode to reduce noise and dilate to focus
    kernel = np.ones((9,9),np.uint8) # 3,3
    mask = cv2.dilate(mask, kernel, iterations=5)
        # mask = cv2.erode(mask, kernel, iterations=5)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #     cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]
    #
    cropped = None

    img_mask = cv2.resize(mask, (600,400))
    cv2.imshow('mask', img_mask)

    # Extract traffic sign
    for i in range(0, len(cnts)):
        cnt = cnts[i]
        x,y,w,h = cv2.boundingRect(cnt)
        offset = 5
        area = w*h

        if w > minW and h > minH and float(h)/w > 0.6 and float(h)/w < 2.0 and area > 600 and area < 30000:
            startY, endY, startX, endX = y,y+h+offset,x,x+w+offset
            if startY > offset:
                startY -= offset
            if startX > offset:
                startX -= offset
            cropped = frame[startY:endY, startX:endX]
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            #resize_img = resize_image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
            #cropped = adjust_gamma(cropped, 2.0)

    if cropped is not None:
        cv2.imshow("crop", cropped)

    return frame, cropped


def boundBox(img):
    out1 = img
    #plt.hist(img.ravel(), 256, [0,256])
    #plt.show()
    #--------------PREPROCESSING------------------
    #Salt and Pepper Noise Removal
    median = cv2.medianBlur(img, 5)
    out1 = np.hstack((out1, median))
    #Histogram Equalization
    median_hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
    equ = cv2.equalizeHist(median_hsv[:, :, 2])
    median_contrast_hsv = median_hsv
    median_contrast_hsv[:, :, 2] = equ
    median_contrast = cv2.cvtColor(median_contrast_hsv, cv2.COLOR_HSV2BGR)
    out2 = median_contrast
    # plt.hist(median_contrast.ravel(), 256, [0,256])
    # plt.show()
    #------------IMAGE PROCESSING-------------------
    new_img = median_contrast
    new_img_hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
    '''new_img_hsv_val = new_img_hsv
    new_img_hsv_val[:, :, 2]'''
    out2 = np.hstack((out2, new_img_hsv))

    #plot H, S, V and histograms for each for all image
    # cv2.imshow('hue', new_img_hsv[:,:,0])
    # cv2.imshow('saturaion', new_img_hsv[:,:,1])
    # cv2.imshow('value', new_img_hsv[:,:,2])
    # histr_hue = cv2.calcHist([new_img_hsv],[0],None,[256],[0,256])
    # plt.plot(histr_hue, color = 'r')
    # plt.xlim([0,256])
    # histr_sat = cv2.calcHist([new_img_hsv],[1],None,[256],[0,256])
    # plt.plot(histr_sat, color = 'g')
    # plt.xlim([0,256])
    # histr_val = cv2.calcHist([new_img_hsv],[2],None,[256],[0,256])
    # plt.plot(histr_val, color = 'b')
    # plt.xlim([0,256])
    # plt.legend(('hue', 'saturation', 'value'))
    # plt.show()


    #Determine ROI
    dimensions = new_img_hsv.shape
    height = img.shape[0]
    width = img.shape[1]
    new_edited_img = np.zeros((height, width, 3), np.uint8)
    new_edited_img_hsv = cv2.cvtColor(new_edited_img, cv2.COLOR_BGR2HSV)
    for i in range(height):
        for j in range(width):
            hue = new_img_hsv[i, j, 0]
            sat = new_img_hsv[i, j, 1]
            val = new_img_hsv[i, j, 2]
            if (hue >= 100 and hue <= 140) and (sat >= 150 and sat<= 255) and (val >= 0 and val<=255):
                new_edited_img_hsv[i, j, 0] = hue
                new_edited_img_hsv[i, j, 1] = sat
                new_edited_img_hsv[i, j, 2] = val
    new_edited_img = cv2.cvtColor(new_edited_img_hsv, cv2.COLOR_HSV2BGR)
    # cv2.imshow('new_edited_img', new_edited_img)

    #thresholding
    new_edited_img_gray = cv2.cvtColor(new_edited_img, cv2.COLOR_BGR2GRAY)
    ret,thresh_img = cv2.threshold(new_edited_img_gray,5,255,cv2.THRESH_BINARY)
    # cv2.imshow('threshold Image', thresh_img)


    img_copy = img
    thresh_img_blur = cv2.GaussianBlur(thresh_img,(7,7),0)
    # cv2.imshow('blur threshold', thresh_img_blur)
    output = cv2.connectedComponentsWithStats(thresh_img_blur, 8, cv2.CV_32S)
    # print(output[0])
    # print(output[1])

    if(len(output[2]) > 1):
        cv2.rectangle(img_copy,(output[2][1][0],output[2][1][1]),(output[2][1][0]+output[2][1][2],output[2][1][1]+output[2][1][3]),(0,255,0),2)
    else:
        return img
    # cv2.imshow('modified',img_copy)

    #plot H, S, V and histograms for each for ROI editted
    # cv2.imshow('hue_edited', new_edited_img_hsv[:,:,0])
    # cv2.imshow('saturaion_edited', new_edited_img_hsv[:,:,1])
    # cv2.imshow('value_edited', new_edited_img_hsv[:,:,2])
    # histr_hue = cv2.calcHist([new_edited_img_hsv],[0],None,[256],[0,256])
    # plt.plot(histr_hue, color = 'r')
    # plt.xlim([0,256])
    # histr_sat = cv2.calcHist([new_edited_img_hsv],[1],None,[256],[0,256])
    # plt.plot(histr_sat, color = 'g')
    # plt.xlim([0,256])
    # histr_val = cv2.calcHist([new_edited_img_hsv],[2],None,[256],[0,256])
    # plt.plot(histr_val, color = 'b')
    # plt.xlim([0,256])
    # plt.legend(('hue', 'saturation', 'value'))
    # plt.show()

    # cv2.imshow('out1', out1)
    # cv2.imshow('out2', out2)
    # cv2.waitKey(0)
    return img_copy
