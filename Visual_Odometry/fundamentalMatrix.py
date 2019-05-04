import numpy as np

def computeFundamentalMatrix(P1, P2):
    A =list()
    n = 1
    for i in range(len(P1)):
        A.append([P1[i][0]*P2[i][0] , P1[i][0]*P2[i][1], P1[i][0], P1[i][1]*P2[i][0], P1[i][1]*P2[i][1], P2[i][1], P2[i][0] , P2[i][1], 1 ])
    A = np.array(A)
    U, s, Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    F = (np.reshape(h, (3, 3)))
    magF = np.linalg.norm(F, np.inf)
    F = F / magF
    FU , Fs, FV = np.linalg.svd(F)
    # print(Fs)
    # Fs[-1] = 0
    Fs = np.array([[Fs[0],0,0],[0,Fs[1],0],[0,0,0]])
    F_hat = np.matmul(np.matmul(FU,Fs),FV)
    return F_hat


def ransac(P1,P2):
    n_iters = 1000
    count = 0
    thres = 5
    ratio = 0.5
    np.random.seed(0)
    while True: #count < n_iters:
        # Computing Random Index
        ranIdx = np.random.randint(len(P1), size=8)
        # Creating three empty list
        ranP1, ranP2, inlinerIdx = list(), list(), list()

        # Points from new and old image from random idx
        for i in range(len(ranIdx)):
            ranP1.append(P1[ranIdx[i]])
            ranP2.append(P2[ranIdx[i]])

        # Compute fundamental Matrix
        F = computeFundamentalMatrix(ranP1, ranP2)

        # Adding Inliners for random index to
        for i in range(len(P1)):
            x_new = np.array([P1[i][0], P1[i][0], 1])
            x_old = np.array([P2[i][0], P2[i][0], 1])
            if abs(x_old @ F @ x_new.T) < thres:
                inlinerIdx.append(i)

        # If ratio of the number of inliners is greater than 0.5 break
        if len(inlinerIdx)/len(P1) > ratio:
            break

    # Creating a inliners P1 and P2
    inlinerP1, inlinerP2 = list(), list()
    for i in range(len(inlinerIdx)):
        inlinerP1.append(P1[i])
        inlinerP2.append(P2[i])

    # updating Fundamental matrix wrt to new points.
    F = computeFundamentalMatrix(inlinerP1, inlinerP2)
    # print(np.linalg.matrix_rank(F))

        # count = count + 1
    return F, inlinerP1, inlinerP2

def computeEssentialMatrix(F):
    K = np.array([ [964.828979, 0,643.788025],[0,964.828979,484.40799 ],[0 ,0, 1] ])
    E = np.matmul(np.matmul(K.T ,F ),K)
    r = np.linalg.matrix_rank(E)
    print(E)
    return E

def estimateCameraPose(E):
    W = np.zeros((3,3))
    W[0][1] = -1
    W[1][0] = 1
    W[2][2] = 1
    U, D, Vt = np.linalg.svd(E)
    Ds = np.array([[D[0],0,0],[0,D[1],0],[0,0,0]])

    C = U[:,2]
    R1 = np.matmul(np.matmul(U,W),Vt)
    R2 = np.matmul(np.matmul(U,W.T),Vt)

    sign = round(np.linalg.det(R1))
    return sign*C, sign*R1, sign*R2
