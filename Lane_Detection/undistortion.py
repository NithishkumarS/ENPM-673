#!/usr/bin/env python3

import numpy as np
import cv2

def get_undistort(img):
    cameraMtx = np.array([[ 1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [  0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
                          [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist = np.array([ -2.42565104e-01 , -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02])
    height, width = img.shape[:2]
    updatedCameraMarix, roi = cv2.getOptimalNewCameraMatrix(cameraMtx, dist,(width,height),0,(width,height))
    undistorted_img = cv2.undistort(img, cameraMtx,dist, None,updatedCameraMarix)
    x,y,w,h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',undistorted_img)
    return undistorted_img 
