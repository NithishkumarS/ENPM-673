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

def superImpose( L_coef, R_coef, h, frame):
    h_inv = np.linalg.inv(h)
    temp = np.zeros_like(frame)
    frame_copy = np.copy(frame)
    if L_coef is None or R_coef is None:
        return frame
    y = np.linspace(350, 719, 720)
    left_x = L_coef[0]*y**2 + L_coef[1]*y + L_coef[2]
    right_x = R_coef[0]*y**2 + R_coef[1]*y + R_coef[2]
    left = np.array([np.transpose(np.vstack([left_x, y]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
    points = np.hstack((left, right))
    cv2.fillPoly(temp, np.int_([points]), (0,255, 0))
    warped_img = cv2.warpPerspective(temp, h_inv, (1280, 720))
    frame = cv2.addWeighted(frame_copy, 1, warped_img, 0.2, 0)
    return frame
