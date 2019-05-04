import numpy as np

def computeFundamentalMatrix(P1, P2):
    A =list()
    n = 1
    for i in range(8):
        A.append([P1[i][0]*P2[i][0] , P1[i][0]*P2[i][1], P1[i][0], P1[i][1]*P2[i][0], P1[i][1]*P2[i][1], P2[i][1], P2[i][0] , P2[i][1], 1 ])
    A = np.array(A)
    U, s, Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    F = (np.reshape(h, (3, 3)))
    magF = np.linalg.norm(F, np.inf)
    F = F / magF
    FU , Fs, FV = np.linalg.svd(F)
    Fs[2][2] = 0
    F_hat = np.matmul(np.matmul(FU,Fs),FV)
    return F_hat


def ransac(P1,P2):
    n_iters = 1000
    count = 0
    thres = 10
    while count < n_iters:
        F = computeFundamentalMatrix()
        count = count + 1
    return F
