import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def least_squares(image, left_lane_hist, right_lane_hist):
    thresh = 130
    binaryImage = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    colorImage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB) 
    x_sum =0
    y_sum = 0
    xy_sum = 0
    x2y_sum = 0
    x4_sum = 0
    x3_sum = 0
    x2_sum = 0
    count = 0
    #cv2.imshow('bw',binaryImage)
    left = [[0,0]]
    right = [[0,0]]
    var = right_lane_hist - left_lane_hist
    for i in range(left_lane_hist-150,left_lane_hist+150):
        for j in range(0,image.shape[0]-1):
            
            if binaryImage[j,i] == 255:
                left = np.concatenate((left,[[i, j]]), axis=0)
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
            if binaryImage[j,i+var] == 255:
                right = np.concatenate((right,[[i+var, j]]), axis=0)

    A = np.array([[x4_sum, x3_sum, x2_sum] , [x3_sum, x2_sum, x_sum], [x2_sum, x_sum, count] ])
    Y = np.array([ [x2y_sum], [xy_sum], [y_sum] ])
    left = np.delete(left, (0,1), axis =0)
    right = np.delete(right, (0,1), axis =0)
    
    #print(count)
    #print(A)
    print('left size:',left.shape)
    print('right size:',right.shape)
    L_coef = np.polyfit(left[:,1].T,left[:,0].T,2)
    R_coef = np.polyfit(right[:,1].T,right[:,0].T,2)
    print('right coeff:',R_coef)
    print('left coeff:',L_coef)
    #    d = np.sqrt(coef[0]*coef[0] + coef[1]*coef[1] + coef[2]*coef[2])
    #print('d',d)
    L_y = np.poly1d(L_coef)
    R_y = np.poly1d(R_coef)
    colorImage2 = colorImage
    #coeff = np.matmul(np.linalg.inv(A), Y)
    '''
    print(coeff[0,0])
    
    print(coeff[1,0])
    print(coeff[2,0])
    '''
    xp = np.linspace(0,1279, 1280)
    L_z = L_y(xp)
    R_z = R_y(xp)
    x = np.zeros(1280)
    for k in range(0,720):
     #   cur = coeff[1,0]*coeff[1,0] - 4*coeff[0,0]*(coeff[2,0] - k)
      #  print('check', cur)
        
       # temp = round(( -coeff[1,0] + math.sqrt(abs(coeff[1,0]*coeff[1,0] - 4*coeff[0,0]*(coeff[2,0] - k)) ))/ (2*coeff[0,0]))
       # x[k] = int(temp)
        colorImage2[k,int(L_z[k])] = [0,0,255]
        colorImage2[k,int(R_z[k])] = [0,0,255]
       # colorImage[k,int(x[k])] = [0,0,255]
       # y[k] = round(coeff[0,0]*k*k + coeff[1,0]*k + coeff[2,0])
      #  colorImage[int(y[k]),k] = (255,0,0)
    
    cv2.imshow('fit', colorImage2)
    return xp,L_z,xp,R_z
#    cv2.imshow('polyfit', colorImage)
