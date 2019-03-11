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
    color_warp = np.zeros_like(frame)
    new_img = np.copy(frame)
    if L_coef is None or R_coef is None:
        return frame
    h = 720
    w = 1280
    ploty = np.linspace(0, h-1, num=h)
    left_fitx = L_coef[0]*ploty**2 + (L_coef[1])*ploty + L_coef[2]
    right_fitx = R_coef[0]*ploty**2 + R_coef[1]*ploty + R_coef[2] +50
    diff = left_fitx[-1] - right_fitx[-1]
    difft = left_fitx[0] - right_fitx[0]
    print(L_coef)
    print(R_coef)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, h_inv, (w, h))
    result = cv2.addWeighted(new_img, 1, np.roll(newwarp,-20,axis=1), 0.2, 0)
    return result
