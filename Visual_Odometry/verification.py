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
from fundamentalMatrix import computeFundamentalMatrix, ransac
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

def norm(pts_new, pts_old):
    pts_new_norm = list()
    pts_old_norm = list()
    for i in range(len(pts_new)):
        pts_new_norm.append([pts_new[i][0],pts_new[i][1]])
        pts_old_norm.append([pts_old[i][0],pts_old[i][1]])
    # pts_new_norm = np.array(pts_new_norm)/np.linalg.norm(pts_new_norm)
    # pts_old_norm = np.array(pts_old_norm)/np.linalg.norm(pts_old_norm)
    return pts_new_norm, pts_old_norm

def computeH(R,t):
    h = np.hstack((R,t))
    h = np.vstack((h, np.array([0,0,0,1])))
    return h

def main():
    """ Main entry point of the app """
    frameCount = 1
    imageList = loadImages()
    old_img = cv2.imread(imageList[0])
    Rc = np.eye(3)
    H = np.eye(4)
    plt.ion()
    Tc = np.zeros((1,3))
    pos = np.zeros((3,1))
    K = np.array([ [964.828979, 0,643.788025],[0,964.828979,484.40799 ],[0 ,0, 1] ])
    while frameCount < len(imageList):
        new_img = cv2.imread(imageList[frameCount])
        pts_new, pts_old = orb(new_img, old_img)
        pts_l_norm, pts_r_norm = norm(pts_new, pts_old)
        E, mask = cv2.findEssentialMat(np.array(pts_l_norm), np.array(pts_r_norm), method=cv2.RANSAC)
        points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm, K)
        print(R)
        print(t)
        # F, pts1, pts2 = ransac(pts_new, pts_old)
        # E = computeEssentialMatrix(F)
        # C, R1, R2 = estimateCameraPose(E)
        # R_final, C_final = triangulation(C, R1, R2, pts1, pts2)
        H = np.matmul(H, computeH(R,t))
        # Rc = np.matmul(Rc, R)
        Tc = H[0:3,3]
        print(Tc)
        # print(R_final)
        # print('C_final:', C_final.shape)
        # pos = np.matmul(R_final,pos) + C_final.reshape((3,1))
        # print(pos)
        plt.plot(Tc[0],Tc[2],'-ro')
        # plt.plot(pos[0][0],pos[2][0],'xr')
        plt.show()
        plt.pause(0.0000001)
        cv2.imshow('frame', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frameCount = frameCount + 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Video extraction
# frame_width = temp.shape[1]
# frame_height = temp.shape[0]
# out = cv2.VideoWriter('Visual_Odometry.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame_width,frame_height))
# out.write(colorImage)
