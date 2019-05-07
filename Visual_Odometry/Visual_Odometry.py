#!/usr/bin/env python3

__author__ = "Nantha Kumar Sunder, Nithish Kumar"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from featureMatch import sift, orb
from fundamentalMatrix import *
from fundamentalMatrix import computeEssentialMatrix, estimateCameraPose
from triangulation import triangulation

def loadImages():
    imageList = []
    for file in os.listdir("Oxford_dataset/stereo/Color"):
        filename, basename = os.path.splitext(file)
        imageList.append(int(filename))
    imageList = sorted(imageList)
    for i in range(len(imageList)):
        imageList[i] = "Oxford_dataset/stereo/Color/" + str(imageList[i]) + ".png"
    return imageList

def computeH(R,t):
    h = np.hstack((R,t))
    h = np.vstack((h, np.array([0,0,0,1])))
    return h

'''
norm
essential matrix

'''
def main():
    """ Main entry point of the app """
    frameCount = 20
    imageList = loadImages()
    
    H = np.eye(4)
    feature_detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    lk_params = dict(winSize=(21, 21),
                     criteria=(cv2.TERM_CRITERIA_EPS |
                     cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    origin = np.zeros((4,1))
    origin[3][0]= 1
    Rc = np.eye(3)
    plt.ion()
    Tc = np.zeros((3,1))    
    old_img = cv2.imread(imageList[frameCount-1])
    old_img = cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
    old_img = cv2.equalizeHist(old_img)
    old_img = cv2.GaussianBlur(old_img,(3,3),0)

    while frameCount < len(imageList):
        prev_keypoint = feature_detector.detect(old_img, None)
        new_img = cv2.imread(imageList[frameCount])
        new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
        new_img = cv2.equalizeHist(new_img)
        new_img = cv2.GaussianBlur(new_img,(3,3),0)
        
        temp = list()
        for i in range(len(prev_keypoint)):
            temp.append([prev_keypoint[i].pt[0], prev_keypoint[i].pt[1]])
        pts_old = np.array(temp, dtype=np.float32)
        pts_new, st, err = cv2.calcOpticalFlowPyrLK(old_img, new_img, pts_old, None, **lk_params)
        print(st.shape)
        st = st.reshape((st.shape[0]))
        pts_new = pts_new[st>0]
        pts_old = pts_old[st>0]
        print('pts_new:', len(pts_new))
        
        F, pts1, pts2 = ransac(pts_new, pts_old)
#        F = computeFundamentalMatrix(pts_new[0:8,:], pts_old[0:8,:])
#         print('F:',F)
        print(len(pts1))
        dd
        E = computeEssentialMatrix(F)
#         print('E:', E)
        
        C, R1, R2 = estimateCameraPose(E)
#         print('C',C)
#         print('R1:',R1)
#         print('R2:',R2)

        R_final, C_final = triangulation(C, R1, R2, pts1, pts2)
#         R_final, C_final = triangulation(C, R1, R2,pts_new, pts_old)
#         print(C_final.T.shape)
#--------------------------------------------------------------------------------------------------
        H = np.matmul(H, computeH(R_final,C_final.T))
        pos = np.matmul(H,origin)
#         print(pos)
        
        plt.plot(pos[0],pos[2],'-ro')
        '''
        Tc = Tc + Rc.dot(C_final.T)
        Rc =Rc.dot(R_final)
    
        plt.plot(Tc[0],Tc[2],'-ro')
        '''
        plt.pause(0.0000001)
#         cv2.imshow('frame', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(frameCount)
        frameCount = frameCount + 1
    plt.show()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Video extraction
# frame_width = temp.shape[1]
# frame_height = temp.shape[0]
# out = cv2.VideoWriter('Visual_Odometry.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
# out.write(colorImage)
