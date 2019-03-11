#!/usr/bin/env python3
import cv2
import numpy as np


def homographicTransform(corner_points, image_points):
    H = cv2.findHomography(np.asmatrix(corner_points), np.asmatrix(image_points))
    return (H)


def getTransfomredImage(h_inv, gray, n_row, n_col):
    h = np.linalg.inv(h_inv)
    im_out = cv2.warpPerspective(gray, h, (n_row,n_col))
    '''
    transformed_image = np.zeros((n_row, n_col), dtype='uint8')
    for row in range(0, n_row):
        for col in range(0, n_col):
            Xc = np.array([col, row, 1]).T
            Xw = np.matmul(h_inv, Xc)
            Xw = (Xw/Xw[2])
            Xw = Xw.astype(int)
            try:
                transformed_image[row][col] = gray[Xw[1]][Xw[0]]
            except:
                pass
    '''
    return im_out

def superImpose(x_points, y_points, h, frame):
    h_inv = np.linalg.inv(h)
    for i in range(0, 719):
        Xc = np.array([x_points[i], y_points[i], 1]).T
        Xw = np.matmul(h_inv, Xc)
        Xw = (Xw/Xw[2])
        Xw = np.round(Xw)
        Xw = Xw.astype(int)
        try:
            frame[Xw[1],Xw[0]-10:Xw[0]+10] = [0,0,255]
        except:
            pass
    return frame
