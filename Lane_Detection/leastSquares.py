import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from werkzeug.formparser import _end
def least_squares(image, left_lane_hist, right_lane_hist,L_coef, R_coef):
    
    prev_L = [0,0,0]
    prev_L = L_coef
    prev_R = [0,0,0]
    prev_R = R_coef
    
    thresh = 130
    binaryImage = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    colorImage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    L_count = 0
    R_count = 0
#     x_sum =0
#     y_sum = 0
#     xy_sum = 0
#     x2y_sum = 0
#     x4_sum = 0
#     x3_sum = 0
#     x2_sum = 0
#     count = 0
    #cv2.imshow('bw',binaryImage)
    left = [[0,0]]
    right = [[0,0]]
    var = right_lane_hist - left_lane_hist
    for i in range(left_lane_hist-150,left_lane_hist+150):
        for j in range(0,image.shape[0]-1):

            if binaryImage[j,i] == 255:
                left = np.concatenate((left,[[i, j]]), axis=0)
                L_count = L_count + 1
                '''
                x_sum = x_sum + i
                y_sum = y_sum + j
                xy_sum = xy_sum + i*j
                x2y_sum = x2y_sum + i*i*j
                x2_sum = x2_sum + i*i
                x3_sum = x3_sum + i*i*i
                x4_sum = x4_sum + i*i*i*i
                count = count +1
                '''
    
    for i in range(right_lane_hist-150,right_lane_hist+150):
        for j in range(0,image.shape[0]-1):
            if binaryImage[j,i] == 255:
                right = np.concatenate((right,[[i, j]]), axis=0)
                R_count = R_count + 1

    print('L_count:',L_count)
    print('R_count:',R_count)
    
    '''
    left = sorted(left, key=lambda left: left[1])
    right = sorted(right, key=lambda right: right[1])
    if L_count > 0: 
        if (L_count) % 2 == 0 :
           median_left =  (left[L_count/2] + left[(L_count-2)/2] ) / 2
        else:
           median_left = left[(L_count-1)/2]
        colorImage[median_left[1]:median_left[1]+20,median_left[0]:median_left[0]+20] = [0,0,255]
        print(median_left)
    else:
        median_left = [0,0]
    if R_count >0:    
        if (R_count) % 2 == 0 :
           median_right =  int( (   right[R_count/2] + right[(R_count-2)/2] ) / 2  )
        else:
           median_right = right[(R_count-1)/2]
        colorImage[median_right[1]:median_right[1]+20,median_right[0]:median_right[0]+20] = [0,0,255]
        print(median_right)
    else:
        median_right = [0,0]
    '''
    '''    
    A = np.array([[x4_sum, x3_sum, x2_sum] , [x3_sum, x2_sum, x_sum], [x2_sum, x_sum, count] ])
    Y = np.array([ [x2y_sum], [xy_sum], [y_sum] ])
    '''
    left = np.delete(left, (0,1), axis =0)
    right = np.delete(right, (0,1), axis =0)

    if R_count > 200 :
        print(1)
        R_coef = np.polyfit(right[:,1].T,right[:,0].T,2)
        print R_coef
    else :
        R_coef = prev_R
   
    if L_count> 200:
        print(2)
        L_coef = np.polyfit(left[:,1].T,left[:,0].T,2)
        print L_coef
    else:
        L_coef = prev_L
        print(3)
    print('Diff', L_coef-R_coef)
    #    if np.absolute(L_coef[0]) > .006 or np.absolute(L_coef[1]) > 1 or np.absolute(L_coef[2]) <100 :
    if np.absolute(R_coef[0] - L_coef[0]) > .01 or np.absolute(R_coef[1] - L_coef[1]) > 1.7 or np.absolute(R_coef[2] - L_coef[2]) > 800:
        L_coef = prev_L
        R_coef = prev_R
        
        
    #    d = np.sqrt(coef[0]*coef[0] + coef[1]*coef[1] + coef[2]*coef[2])
    #print('d',d)
    L_y = np.poly1d(L_coef)
    R_y = np.poly1d(R_coef)
    L_end = L_y(719)
    R_end = R_y(719)
    print('L_end - R_end',L_end - R_end)
    if np.absolute(L_end - R_end) < 480 and np.absolute(L_end - R_end) < 520 or L_end > 600:
         L_coef = prev_L
         R_coef = prev_R
    
    L_y = np.poly1d(L_coef)
    R_y = np.poly1d(R_coef)
    colorImage2 = colorImage
    #coeff = np.matmul(np.linalg.inv(A), Y)
    
    xp = np.linspace(0,1279, 1280)
    L_z = L_y(xp)
    R_z = R_y(xp)
    x = np.zeros(1280)
    for k in range(0,720):
     #   cur = coeff[1,0]*coeff[1,0] - 4*coeff[0,0]*(coeff[2,0] - k)
      #  print('check', cur)

       # temp = round(( -coeff[1,0] + math.sqrt(abs(coeff[1,0]*coeff[1,0] - 4*coeff[0,0]*(coeff[2,0] - k)) ))/ (2*coeff[0,0]))
       # x[k] = int(temp)
       try:
           colorImage2[k,int(L_z[k])] = [0,0,255]
           colorImage2[k,int(R_z[k])] = [0,0,255]
       except:
           pass
       # colorImage[k,int(x[k])] = [0,0,255]
       # y[k] = round(coeff[0,0]*k*k + coeff[1,0]*k + coeff[2,0])
      #  colorImage[int(y[k]),k] = (255,0,0)
    
    cv2.imshow('fit', colorImage2)
    print('L_coef:', L_coef)
    print('R_coef:', R_coef)
    
    return xp,L_z,xp,R_z, L_coef, R_coef
   #    cv2.imshow('polyfit', colorImage)
