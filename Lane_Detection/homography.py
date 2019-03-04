#!/usr/bin/env python3
import cv2
import numpy as np


def homographicTransform(corner_points, image_points):
    H = cv2.findHomography(np.asmatrix(corner_points), np.asmatrix(image_points))
    return (H)


def getTransfomredImage(h_inv, gray, n):
    transformed_image = np.zeros((n, n), dtype='uint8')
    for row in range(0, n):
        for col in range(0, n):
            Xc = np.array([col, row, 1]).T
            Xw = np.matmul(h_inv, Xc)
            Xw = (Xw/Xw[2])
            Xw = Xw.astype(int)
            transformed_image[row][col] = gray[Xw[1]][Xw[0]]
    return transformed_image
