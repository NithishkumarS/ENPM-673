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
from scipy.stats import multivariate_normal as mvn

def get_thresholded_pdf(frame, p1):
    frame[p1 == True] = 255
    frame[p1 == False] = 0
    return frame

def draw_buoy_contour(original_frame, reference_frame, color, prev_centroid):
    contours, hier = cv2.findContours(reference_frame, 1, 2)
    radius_r = []
    if contours:
        for c in contours:
            point,radius = cv2.minEnclosingCircle(c)
            radius_r.append(int(radius))
        max_r = np.argmax(radius_r)
        cnt = contours[max_r]
        moments = [cv2.moments(cnt)]
        centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]
        print(prev_centroid)
        for c in centroids:
            if not prev_centroid:
                cv2.circle(original_frame, c, radius_r[max_r], color, thickness=2)
            elif  np.sqrt((prev_centroid[0][1] - c[1])**2 + (prev_centroid[0][0] - c[0])**2) < 500:
                cv2.circle(original_frame, c, radius_r[max_r], color, thickness=2)
            else:
                return original_frame, prev_centroid
            # else:
            #     print('dist: ',np.sqrt((prev_centroid[0][1] - c[1])**2 + (prev_centroid[0][0] - c[0])**2) )
            #     cv2.circle(original_frame, c, radius_r[max_r], color, thickness=2)



    return original_frame, centroids

def detectTuning(log_likelihood, img, i):
    #log_likelihood = get_thresholded_pdf(log_likelihood, log_likelihood > 0.65*np.max(log_likelihood))
    if (i==0):
        kernel = np.ones((7,7),np.uint8)
        thresh = np.max(log_likelihood)
        print(0.5*np.max(log_likelihood))
        op = np.bitwise_and(log_likelihood > 0.5*thresh, log_likelihood < .000025)
        log_likelihood = get_thresholded_pdf(log_likelihood, (op))
        log_likelihood = cv2.dilate(log_likelihood,kernel,iterations = 1)
    else:
        log_likelihood = get_thresholded_pdf(log_likelihood, log_likelihood > 0.5*np.max(log_likelihood))

    return log_likelihood

def detectbuoy(img,prev_centroid):
    K = 4
    colors = ['Red', 'Yellow', 'Green']
    colors_value = [(0,0,255),(0,255,255),(0,255,0)]
    # prev_centroid = []
    for i in range(1,2):
        w=np.load('weights_' +colors[i] + '.npy')
        Sigma=np.load('sigma_' +colors[i] + '.npy')
        mean=np.load('mean_' +colors[i] + '.npy')
        nr, nc, d = img.shape
        n=nr*nc
        xtest=np.reshape(img,(n,d))
        likelihoods=np.zeros((K,n))
        log_likelihood=np.zeros(n)
        for k in range(K):
            likelihoods[k] = w[k] * mvn.pdf(xtest, mean[k], Sigma[k],allow_singular=True)
            log_likelihood = likelihoods.sum(0)
        log_likelihood = np.reshape(log_likelihood, (nr, nc))
        #log_likelihood[log_likelihood > 0.65*np.max(log_likelihood) ] = 255
        #log_likelihood = get_thresholded_pdf(log_likelihood, log_likelihood > 0.65*np.max(log_likelihood))
        log_likelihood = detectTuning(log_likelihood, img, i)
        log_likelihood = log_likelihood.astype(np.uint8)

        #log_likelihood = cv2.dilate(log_likelihood,kernel,iterations = 1)
        frame, prev_centroid = draw_buoy_contour(img, log_likelihood, colors_value[i], prev_centroid)
    return log_likelihood, prev_centroid

def main():
    cap = cv2.VideoCapture("detectbuoy.avi")
    centroid = []
    while cap.isOpened():
        ret, img = cap.read()
        frame, centroid = detectbuoy(img, centroid)
        cv2.imshow('Buoy Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
