#!/usr/bin/env python3
import cv2
import numpy as np


def homographicTransform(cornerPoints):
#corner):
    contourCount = len(cornerPoints) / 4
    #print(cornerPoints)
    if contourCount == 1:
        Xw = cornerPoints #np.concatenate((cornerPoints[corner[0]], cornerPoints[corner[1]],
             # cornerPoints[corner[2]], cornerPoints[corner[3]]), axis=0)
        #print('Xw')
        #print(Xw)
        #print(np.asmatrix(Xw))

    Xc = np.array([[0, 0], [199, 0], [199, 199], [0, 199]])
    #Xc = np.float32([[2,3],[2,5],[6,3],[6,5]])
    A = np.zeros((1, 9), int)
    temp = np.zeros((1, 9), int)
    temp1 = np.zeros((1, 9), int)
    n = 1
    for i in range(0, 4):
     #   print(i)
        '''
        temp = np.array([Xw[i][0], Xw[i][1], 1, 0, 0, 0, (-Xc[i][0]) *
                         (Xw[i][0]), (-Xc[i][0])*(Xw[i][1]), -Xc[i][0]])
        temp1 = np.array([0, 0, 0, Xw[i][0], Xw[i][1], 1, (-Xc[i][1]) *
                          (Xw[i][0]), (-Xc[i][1])*(Xw[i][1]), -Xc[i][1]])
        '''
        temp = np.array([Xw[i][0], Xw[i][1], 1, 0, 0, 0, (-Xc[i][0]) *
                         (Xw[i][0]), (-Xc[i][0])*(Xw[i][1]), -Xc[i][0]])
        temp1 = np.array([0, 0, 0, Xw[i][0], Xw[i][1], 1, (-Xc[i][1]) *
                          (Xw[i][0]), (-Xc[i][1])*(Xw[i][1]), -Xc[i][1]])
        temp = np.append([temp], [temp1], axis=0)
        # A = np.append(A, temp, axis=0)

        # print(temp)
        if n == 1:
            A = temp
            n = 2
        else:
            A = np.append(A, temp, axis=0)

    print(A)
    U, s, V = np.linalg.svd(A)

    h = V[-1,:]/V[-1,-1]
    print(h)
    H = (np.reshape(h, (3, 3)))
    # H = H/H[2][2]
    print('H from cal')
    print(H)
    print(np.linalg.inv(H))
    H_cv, status = cv2.findHomography(np.asmatrix(Xw), np.asmatrix(Xc))
    print('homography from function')
    print(H_cv)
    print(np.linalg.inv(H_cv))
    return (H)
    #H1 = cv2.getPerspectiveTransform(Xw, Xc)

    #img = cv2.imread('ref_marker.png')
    #height, width, channels = img.shape
    # print(width)
    # print(height)

    #dst = cv2.warpPerspective(img,H,(200,200))
    #processed_img = np.zeros(10,10)
    #cv2.imshow('transformed Image', dst)
    #cv2.imshow( "Display window", dst )
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
