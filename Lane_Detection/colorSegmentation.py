#!/usr/bin/env python3

import numpy as np
import cv2

def colorSegmentation(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #cv2.imshow('equalizeHist', img_output)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_output = clahe.apply(gray)

    thres = [20, 255];
    image_hls = cv2.cvtColor(image,  cv2.COLOR_BGR2HLS).astype(np.float)

    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper).astype(np.uint8)
    image_white_mask = cv2.bitwise_and(image, image, mask = white_mask)
    bi_im_whi = convert2Bin(image_white_mask, thres)

    # yellow color mask
    lower = np.uint8([ 20, 120, 80])
    upper = np.uint8([ 45, 200, 255])
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(image, lower, upper).astype(np.uint8)
    image_yellow_mask = cv2.bitwise_and(image, image, mask = yellow_mask)
    bi_im_yel_hsv = convert2Bin(image_yellow_mask, thres)

    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper).astype(np.uint8)
    image_yellow_mask = cv2.bitwise_and(image, image, mask = yellow_mask)
    bi_im_yel = convert2Bin(image_yellow_mask, thres)


    # yellow color mask hls
    lower = np.uint8([ 20, 120, 80])
    upper = np.uint8([ 45, 200, 255])
    yellow_mask_hls = cv2.inRange(image_hls, lower, upper).astype(np.uint8)
    image_yellow_mask_hls = cv2.bitwise_and(image, image, mask = yellow_mask_hls)
    temp = cv2.cvtColor(image_yellow_mask_hls, cv2.COLOR_HLS2BGR)
    bi_im_yel_hls = convert2Bin(temp, thres)

    #
    hls_l = image_hls[:,:,1]
    hls_l = hls_l*(255/np.max(hls_l))
    thres = [220, 255];
    yellow_mask_hls = cv2.inRange(hls_l, 220, 255).astype(np.uint8)
    temp = cv2.cvtColor(image_yellow_mask_hls, cv2.COLOR_HLS2BGR)
    bi_im_yel_hls_2 = convert2Bin(temp, thres)

    # combine the mask
    binary_image = np.zeros_like(bi_im_yel_hls)
    binary_image = cv2.bitwise_or(bi_im_yel_hls, bi_im_whi)
    binary_image = cv2.bitwise_or(binary_image, bi_im_yel)
    binary_image = cv2.bitwise_or(binary_image, bi_im_yel_hls_2)


    return binary_image, img_output

def convert2Bin(image, thres):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.threshold(gray, thres[0], thres[1], cv2.THRESH_BINARY)[1]
    return binary_image
