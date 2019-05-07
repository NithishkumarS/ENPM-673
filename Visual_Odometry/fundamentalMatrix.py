import numpy as np

def normalize(points):
    point = points.copy()
#     print('shape:',point.T)
    mean = np.mean(point, axis=0)
#     print('mean:',mean.shape)
    pointCen = point - mean

    meanDist = np.mean(np.sqrt(np.sum(pointCen**2, axis=1)))

    if meanDist > 0:
        scale = np.sqrt(2)/meanDist
    else:
        scale = 1

    scaleMat = np.array([[scale,0,-scale*mean[0]],[0,scale,-scale*mean[1]],[0,0,1]])
    normalizedPoint = np.matmul(scaleMat, point.T).T
    return normalizedPoint, scaleMat

def computeFundamentalMatrix(pts_new, pts_old):
    A =list()
    n = 1
    for i in range(len(pts_new)):
        A.append([pts_new[i][0]*pts_old[i][0] , pts_new[i][0]*pts_old[i][1], pts_new[i][0], pts_new[i][1]*pts_old[i][0], pts_new[i][1]*pts_old[i][1], pts_old[i][1], pts_old[i][0] , pts_old[i][1], 1 ])
    A = np.array(A)
    _ , _ , Vt = np.linalg.svd(A)
    h = Vt[-1,:]
    F = (np.reshape(h, (3, 3)))
    # magF = np.linalg.norm(F, np.inf)
    # F = F / magF
    # print(F)
    FU , Fs, FV = np.linalg.svd(F)
    
#     Fs = np.array([[(Fs[0]+Fs[2])/2,0,0],[0,(Fs[1]+Fs[2])/2,0],[0,0,0]])
    
    Fs = np.array([[Fs[0],0,0],[0,Fs[1],0],[0,0,0]])
    
    F_hat = np.matmul(np.matmul(FU,Fs),FV)

#     F_hat = F_hat / np.linalg.norm(F_hat)
    F_hat = F_hat / F_hat[-1][-1]
    if F_hat[-1][-1] < 0:
        F_hat = -F_hat
    return F_hat


def ransac(pts_new,pts_old):
    pts_new = np.hstack((pts_new, np.ones((len(pts_new), 1))))
    pts_old = np.hstack((pts_old, np.ones((len(pts_old), 1))))
#     pts_new, oldT = normalize(pts_new)
#     pts_old, newT = normalize(pts_old)
  
    n_iters = 1000
    count = 0
    n = 0
    thres = 0.5
    # ratio =
    np.random.seed(0)
    while count < n_iters: #count < n_iters:
        # Computing Random Index
        ranIdx = np.random.randint(len(pts_new), size=8)
        # Creating three empty list
        ranP1, ranP2, inlinerIdx = list(), list(), list()

        # Points from new and old image from random idx
        for i in range(len(ranIdx)):
            ranP1.append(pts_new[ranIdx[i]])
            ranP2.append(pts_old[ranIdx[i]])

        ranP1 = np.array(ranP1)
        ranP2 = np.array(ranP2)

        # Compute fundamental Matrix
        F = computeFundamentalMatrix(ranP1, ranP2)

        # Adding Inliners for random index to
        for i in range(len(pts_new)):
            x_new = np.array([pts_new[i][0], pts_new[i][1], 1])
            x_old = np.array([pts_old[i][0], pts_old[i][1], 1])
            if abs(np.matmul(x_old,np.matmul(F,x_new.T)) ) < thres:
                inlinerIdx.append(i)

        if n < len(inlinerIdx):
            finalIdx = inlinerIdx
            n = len(inlinerIdx)
        # # If ratio of the number of inliners is greater than 0.5 break
        # if len(inlinerIdx)/len(P1) > ratio:
        #     break

        count = count + 1
    # Creating a inliners P1 and P2
    inlinerP1, inlinerP2 = list(), list()
    for i in range(len(finalIdx)):
        inlinerP1.append(pts_new[finalIdx[i]])
        inlinerP2.append(pts_old[finalIdx[i]])
    print(len(inlinerP1))
    print(len(pts_new))
    inlinerP1 = np.array(inlinerP1)
    inlinerP2 = np.array(inlinerP2)
    # updating Fundamental matrix wrt to new points.
    F = computeFundamentalMatrix(inlinerP1, inlinerP2)
    # print(np.linalg.matrix_rank(F))
#     F = np.matmul(np.matmul(newT.T,F),oldT)
    print('F after',F/F[-1][-1])
    
        # count = count + 1
    return F, inlinerP1, inlinerP2

def computeEssentialMatrix(F):
    K = np.array([ [964.828979, 0,643.788025],[0,964.828979,484.40799 ],[0 ,0, 1] ])
    E = np.matmul(np.matmul(K.T ,F ),K)
    r = np.linalg.matrix_rank(E)
    u, d, v = np.linalg.svd(E)
    d[-1] = 0
    #    makes singular values 1
    s = np.array([[d[0],0,0],[0,d[1],0],[0,0,0]])
#     s = np.array([[1,0,0],[0,1,0],[0,0,0]])
    E_hat = np.matmul(u,np.matmul(s,v))
    print(E_hat)
    ff
    return E_hat

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
