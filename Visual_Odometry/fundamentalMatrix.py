import numpy as np

def computeFundamentalMatrix(P1, P2):
    A =list()
    n = 1
    for i in range(8):
        A.append([P1[i][0]*P2[i][0] , P1[i][0]*P2[i][1], P1[i][0], P1[i][1]*P2[i][0], P1[i][1]*P2[i][1], P2[i][1], P2[i][0] , P2[i][1], 1 ])
    A = np.array(A)
    U, s, V = np.linalg.svd(A)
    h = V[-1,:]/V[-1,-1]
    H = (np.reshape(h, (3, 3)))
    return H

def RANSAC(P1,P2):
    n_iters = 1000
    count = 0
    thres = 10
    while count < n_iters:
        F = computeFundamentalMatrix()
        count = count + 1
    return F
