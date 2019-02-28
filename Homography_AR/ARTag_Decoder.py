#! /usr/bin/env python3

import os
import sys

import cv2
import numpy as np


def decode(A):
    """Find ID and orientation of the AR Tag"""
    ID = 0
    m = 0
    n = 0
    M = int(A.shape[0]/8)
    N = int(A.shape[1]/8)
    mm = m + M
    nn = n + N
    mean = np.zeros([8, 8], dtype=int)
    row = 0
    col = 0
    threshold = 220
    # Converting to 8*8 from m*n based on given AR Tag
    while(mm <= A.shape[0] and nn <= A.shape[1]):
        nn = n + N
        a = A[m:mm, n:nn]
        # print('m(%d):mm(%d)   |   n(%d):nn(%d)' % (m, mm, n, nn))
        mean_a = np.mean(a)
        mean[row, col] = 0 if mean_a < threshold else 1
        n = n + N
        col += 1
        if nn >= A.shape[1]:
            row += 1
            col = 0
            m = m + M
            mm = mm + M
            n = 0
    for i in range(0, 4):
        rotated_mean = np.rot90(mean, i)
        #print('rotated mean')
        #print(rotated_mean)

        #print(rotated_mean[5, 5])
        # Detecting orientation based on reference
        if int(rotated_mean[5, 5]) == 1:
            #print(rotated_mean[3, 3])
            binary = '%s%s%s%s' % (rotated_mean[3, 3], rotated_mean[3, 4],
                                   rotated_mean[4, 3], rotated_mean[4, 4])
            #print(binary)
            ID = int(binary, 2)
            break
    #print(i)
    return ID,i
