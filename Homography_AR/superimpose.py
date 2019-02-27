import cv2
import numpy as np
def superImpose(h_inv, frame,rotations):

    img = cv2.imread('Reference Images/Lena.png',1)
    print(type(img))
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 90*rotations, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    # print(result.shape)

    ##cv2.imshow('Normal', frame)
    for row in  range(0,512):
        for col in range(0,512):
            X_dash = np.array([col,row,1]).T
            X = np.matmul(h_inv,X_dash)
            X = (X/X[2])
            X = X.astype(int)
            frame[X[1]][X[0]] = result[col][row]

    # cv2.imshow('Superimposed', frame)
    #cv2.waitKey(1)
    len(img)
    return frame
