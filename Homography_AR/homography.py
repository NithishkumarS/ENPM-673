#!/usr/bin/env python3
import cv2
import numpy as np


def homographicTransform(cornerPoints, Xc):
    contourCount = len(cornerPoints) / 4
    if contourCount == 1:
        Xw = cornerPoints
    A = np.zeros((1, 9), int)
    temp = np.zeros((1, 9), int)
    temp1 = np.zeros((1, 9), int)
    n = 1
    for i in range(0, 4):
        temp = np.array([Xw[i][0], Xw[i][1], 1, 0, 0, 0, (-Xc[i][0]) *
                         (Xw[i][0]), (-Xc[i][0])*(Xw[i][1]), -Xc[i][0]])
        temp1 = np.array([0, 0, 0, Xw[i][0], Xw[i][1], 1, (-Xc[i][1]) *
                          (Xw[i][0]), (-Xc[i][1])*(Xw[i][1]), -Xc[i][1]])
        temp = np.append([temp], [temp1], axis=0)
        if n == 1:
            A = temp
            n = 2
        else:
            A = np.append(A, temp, axis=0)
    U, s, V = np.linalg.svd(A)
    h = V[-1,:]/V[-1,-1]
    H = (np.reshape(h, (3, 3)))
    return (H)
