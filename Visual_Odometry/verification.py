#!/usr/bin/env python3
__author__ = "Nantha Kumar Sunder, Nithish Kumar, Rama Prashanth"
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
from fundamentalMatrix import computeFundamentalMatrix, ransac, normalize
from fundamentalMatrix import computeEssentialMatrix, estimateCameraPose
from triangulation import triangulation

def masker(img):
    img[int(img.shape[0]*3/4):,:] = 0
    return img

def loadImages():
    imageList = []
    for file in os.listdir("Oxford_dataset/stereo/Color"):
        filename, basename = os.path.splitext(file)
        imageList.append(int(filename))
    imageList = sorted(imageList)
    for i in range(len(imageList)):
        imageList[i] = "Oxford_dataset/stereo/Color/" + str(imageList[i]) + ".png"
    return imageList

def norm(pts_new, pts_old):
    pts_new_norm = list()
    pts_old_norm = list()
    for i in range(len(pts_new)):
        pts_new_norm.append([pts_new[i][0],pts_new[i][1]])
        pts_old_norm.append([pts_old[i][0],pts_old[i][1]])
#     pts_new_norm = np.array(pts_new_norm)/np.linalg.norm(pts_new_norm)
#     pts_old_norm = np.array(pts_old_norm)/np.linalg.norm(pts_old_norm)
    return pts_new_norm, pts_old_norm

def computeH(R,t):
    h = np.hstack((R,t))
    h = np.vstack((h, np.array([0,0,0,1])))
    # print('h:',h)
    # print('R:',R)
    # print('t:',t)
    return h

def main():
    """ Main entry point of the app """
    frameCount = 20
    imageList = loadImages()
    H = np.eye(4)
    plt.ion()
    feature_detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    lk_params = dict(winSize=(21, 21),
                    maxLevel=3,
                    minEigThreshold=0.001,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                     cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    pos = np.zeros((3,1))
    K = np.array([ [964.828979, 0,643.788025],[0,964.828979,484.40799 ],[0 ,0, 1] ])
    R_f = np.eye(3)
    T_f = np.zeros((3,1))
    origin = np.zeros((4,1))
    origin[3][0]= 1
    # print(origin)

    while frameCount < len(imageList):
        old_img = cv2.imread(imageList[frameCount-5])
        old_img = cv2.cvtColor(old_img,cv2.COLOR_BGR2GRAY)
        old_img = cv2.equalizeHist(old_img)
        old_img = cv2.GaussianBlur(old_img,(3,3),0)
        old_img = masker(old_img)
        prev_keypoint = feature_detector.detect(old_img, None)
        new_img = cv2.imread(imageList[frameCount])
        new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
        new_img = cv2.equalizeHist(new_img)
        new_img = cv2.GaussianBlur(new_img,(3,3),0)
        new_img = masker(new_img)

        temp = list()
        for i in range(len(prev_keypoint)):
            temp.append([prev_keypoint[i].pt[0], prev_keypoint[i].pt[1]])
        points = np.array(temp, dtype=np.float32)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, new_img, points, None, **lk_params)
        st = st.reshape(st.shape[0])
        p1 = p1[st>0]
        points = points[st>0]
        # pts_new, pts_old = orb(new_img, old_img)
        '''
        print(pts_new.shape,pts_old.shape )
        cv2.line(old_img,(int(pts_old[30][0]),int(pts_old[30][1])),(int(pts_old[21][0]),int(pts_old[21][1])),(255,0,0),5)
        cv2.line(new_img,(int(pts_new[30][0]),int(pts_new[30][1])),(int(pts_new[21][0]),int(pts_new[21][1])),(255,0,0),5)
        cv2.imshow('old:', old_img)
        cv2.imshow('new', new_img)
        cv2.waitKey(0)
        '''
#         fundamental_matrix,mask1 = cv2.findFundamentalMat(np.array(pts_l_norm), np.array(pts_r_norm), cv2.FM_RANSAC, 1, 0.99);
#         print('FUnd',fundamental_matrix)
##---------------------------------------------------------------------------------------
        E, mask = cv2.findEssentialMat(p1, points, K, cv2.RANSAC ,0.999, 1.0, None)
        mask = mask.reshape(mask.shape[0])
        p1 = p1[mask>0]
        points = points[mask>0]
        # print('E:', E)
        points, R, t, mask = cv2.recoverPose(E, p1, points, K)
        # print(R)
        # print(t)
##---------------------------------------------------------------------------------------

        H = np.matmul(H, computeH(R,t))
        pos = np.matmul(H,origin)
#         print(pos)
        # print(pos)

        plt.plot(pos[0],pos[2],'-ro')
        plt.show()
        plt.pause(0.0000001)

        cv2.imshow('frame', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        old_img = new_img.copy()
        frameCount = frameCount + 5
        print(frameCount)
        # old_img = new_img.copy()
        # p0 = good_new.reshape(-1,1,2)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Video extraction
# frame_width = temp.shape[1]
# frame_height = temp.shape[0]
# out = cv2.VideoWriter('Visual_Odometry.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
# out.write(colorImage)
