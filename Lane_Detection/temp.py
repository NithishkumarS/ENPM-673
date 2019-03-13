import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def least_squares(image, left_lane_hist, right_lane_hist,L_coef, R_coef):
    prev_L = L_coef
    prev_R = R_coef
    thresh = 130
    image[0:int(image.shape[0]*1/3),:] = 0
    binaryImage = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    colorImage = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    diff = right_lane_hist - left_lane_hist
    L_count = 0
    R_count = 0
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
            if binaryImage[j,i+var] == 255:
                right = np.concatenate((right,[[i+var, j]]), axis=0)
                R_count = R_count + 1
#     median_left= [0,0]
#     median_right = [0,0]
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

    print('left size:',left.shape)
    print('right size:',right.shape)


    if R_count > 50 :
        R_coef = np.polyfit(right[:,1].T,right[:,0].T,2)
    else:
        R_coef = prev_R
    if L_count> 50:
        L_coef = np.polyfit(left[:,1].T,left[:,0].T,2)
    else:
        L_coef = prev_L
    print('R_coef', R_coef)
    print('L_coef', L_coef)
    print('prev_R', prev_R)
    print('prev_L', prev_L)
    ###################################################################
    # IF all of previous git is zero then pre fit is present findContours
    if (np.all(prev_R) == 0 and np.all(prev_L) == 0 ):
        R_coef = np.polyfit(right[:,1].T,right[:,0].T,2)
        L_coef = np.polyfit(left[:,1].T,left[:,0].T,2)
        prev_R = R_coef
        prev_L = L_coef

    # print('right coeff:',R_coef)
    # print('left coeff:',L_coef)
    #    d = np.sqrt(coef[0]*coef[0] + coef[1]*coef[1] + coef[2]*coef[2])
    #print('d',d)
    L_y = np.poly1d(L_coef)
    R_y = np.poly1d(R_coef)
    colorImage2 = colorImage

    #coeff = np.matmul(np.linalg.inv(A), Y)

    xp = np.linspace(0,1279, 1280)
    L_z = L_y(xp)
    R_z = R_y(xp)
    ################################
    ## bottom calculation
    y_max_right = np.argmax(xp)
    x_max_right = R_z[y_max_right]
    print(y_max_right)
    print(x_max_right)

    y_max_left = np.argmax(xp)
    x_max_left = L_z[y_max_left]
    print(y_max_left)
    print(x_max_left)
    diff2_bot = x_max_right - x_max_left
    print('histo diff',diff)
    print('Curve diff',diff2_bot)
    lateral_dist_bot = diff - diff2_bot
    print('lateral_bot',lateral_dist_bot)
    ######################
    ## top calculation
    y_min_right = np.argmin(xp)
    x_min_right = R_z[y_min_right]
    print(y_min_right)
    print(x_min_right)

    y_min_left = np.argmin(xp)
    x_min_left = L_z[y_min_left]
    print(y_min_left)
    print(x_min_left)
    diff2_top = x_min_right - x_min_left
    print('histo diff',diff)
    print('Curve diff',diff2_top)
    lateral_dist_top = diff - diff2_top
    print('lateral_top',lateral_dist_top)
    if (left_lane_hist < 300 or left_lane_hist > 400):
        L_coef = prev_L
    if (left_lane_hist < 820 or left_lane_hist > 950):
        L_coef = prev_L

    if (np.abs(lateral_dist_top) > 150):
        '''
        if (x_max_right < x_max_left):
            R_coef = prev_R
        else:
            R_coef = prev_R
            L_coef = prev_L
        '''
        R_coef = prev_R
        L_coef = prev_L
        #R_coef
        #prev_R

    #################################3

    print('left hist', left_lane_hist)
    print('right hist', right_lane_hist)
    '''
    x = np.zeros(1280)
    for k in range(0,720):
     #   cur = coeff[1,0]*coeff[1,0] - 4*coeff[0,0]*(coeff[2,0] - k)
      #  print('check', cur)

       # temp = round(( -coeff[1,0] + math.sqrt(abs(coeff[1,0]*coeff[1,0] - 4*coeff[0,0]*(coeff[2,0] - k)) ))/ (2*coeff[0,0]))
       # x[k] = int(temp)
       try:
     #      colorImage2[k,int(L_z[k])] = [0,0,255]
           colorImage2[k,int(R_z[k])] = [0,0,255]
       except:
           pass
       # colorImage[k,int(x[k])] = [0,0,255]
       # y[k] = round(coeff[0,0]*k*k + coeff[1,0]*k + coeff[2,0])
      #  colorImage[int(y[k]),k] = (255,0,0)
    '''
    #cv2.imshow('fit', colorImage2)

    return xp,L_z,xp,R_z, L_coef, R_coef
   #    cv2.imshow('polyfit', colorImage)
