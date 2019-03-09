#!/usr/bin/env python3

import numpy as np
import cv2
import matplotlib.pyplot as plt
def polyfit(img, ll_pt, rl_pt):

    return img

def slidingWindowFit(img, ll_pt, rl_pt):
    win_size_x = 20
    win_size_y = 10
    image_size = img.shape
    y_pos = 399
    gray = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mean_point = np.zeros((1,2))
    # left line
    for i in range(0,40):
        rect_top = np.array([ll_pt - win_size_x, y_pos - win_size_y])
        rect_bot = np.array([ll_pt + win_size_x, y_pos])
        # finding the mean
        temp_point = np.argmax(np.mean(gray[rect_top[1]:rect_bot[1],rect_top[0]:rect_bot[0]], axis=0)) + rect_top[0] - 1
        if np.mean(gray[y_pos-win_size_y:y_pos, temp_point]) > 150:
            temp_array = np.array([[temp_point, y_pos]])
            print(temp_array.shape)
            print(mean_point.shape)
            mean_point=np.concatenate((mean_point, temp_array), axis=0)
            diff = ll_pt - temp_point
            ll_pt = ll_pt - diff
            rect_top = np.array([ll_pt - win_size_x, y_pos - win_size_y])
            rect_bot = np.array([ll_pt + win_size_x, y_pos])
        cv2.rectangle(img, (rect_top[0], rect_top[1]), (rect_bot[0], rect_bot[1]), (0,0,255), 0)
        y_pos = y_pos - win_size_y - 1
    
    print('here')
    
    mean_point = np.delete(mean_point, (0,1), axis =0)
    print('meanpoint')
    print(mean_point[:,0])
    
    coef = np.polyfit(mean_point[:,0].T,mean_point[:,1].T,2)
    #    d = np.sqrt(coef[0]*coef[0] + coef[1]*coef[1] + coef[2]*coef[2])
    #print('d',d)
    size = mean_point.shape[1]
    
    y = np.poly1d(coef)
    print(mean_point[size,0])
    xp = np.linspace(mean_point[0,0], mean_point[size,0], 400)
    print(xp)
    z = y(xp)
    print(z)
    plt.plot(xp,z)
    plt.show()


    cv2.imshow('Rectangle', img)
    return img
