#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect_turn(L_coef, R_coef, image_shape, l_coef_arr, r_coef_arr):
# =============================================================================
#     # Dimensions along x & y in pixels
#     xmpix = 30/720  #30 meters is equivalent to 720 pixels in the vertical direction
#     ympix = 3.7/700 #3.7 meters is equal to 700 pixels in the horizontal direction
#     yindex = 719  # 720 pixel; last y index = 719
#     # Fit x,y in world space
#     lwfit = np.polyfit(lefty*ympix, leftx*xmpix, 2)
#     rwfit = np.polyfit(righty*ympix, rightx*xmpix, 2)
#     # R curvature in meters
#     lcurv = ((1 + (2*lwfit[0]*yindex*ympix + lwfit[1])**2)**1.5) / np.absolute(2*lwfit[0])
#     rcurv = ((1 + (2*rwfit[0]*yindex*ympix + rwfit[1])**2)**1.5) / np.absolute(2*rwfit[0])
#     curv =  (lcurv + rcurv) / 2
#     print('Left Curvature ', lcurv)
#     print('Right Curvature ', rcurv)
#     print('Curvature ', curv)
#
#
#     # Calculate vehicle center offset in pixels
#     left_fit = np.polyfit(lefty, leftx, 2)
#     right_fit = np.polyfit(righty, rightx, 2)
#     bottom_y = image_shape[0] - 1
#     print('Left fit',left_fit)
#     bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
#     bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
#     xmean = (bottom_x_left + bottom_x_right) / 2
#     # +ve offset --> right | -ve offset --> left
#     offset = (image_shape[1]/2 - xmean) * xmpix
#     print('Offset ',offset)
#     print('L Coef', L_coef)
#     print('R Coef', R_coef)
# =============================================================================

    l_coef_arr.append([L_coef[0]])
    r_coef_arr.append([R_coef[0]])
    if (len(l_coef_arr) >= 7):
        l_coef_arr.pop(0)
        r_coef_arr.pop(0)

    l_mean_coeff = np.mean(l_coef_arr)
    r_mean_coeff = np.mean(r_coef_arr)

    print('Left Mean Coeff', l_mean_coeff)
    print('Right Mean Coeff', r_mean_coeff)
    turn = 'Straight'
    if (l_mean_coeff < -0.00009) and (r_mean_coeff < -0.00009):
        print('Turn Left')
        turn = 'Turn Left'
    elif (l_mean_coeff > 0.00009) and (r_mean_coeff > 0.00009):
        print('Turn Right')
        turn = 'Turn Right'

    return turn, l_coef_arr, r_coef_arr
