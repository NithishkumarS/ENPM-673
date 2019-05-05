import numpy as np

def triangulation(C0, R1, R2, pts2, pts1):
    K = np.array([ [964.828979, 0,643.788025],[0,964.828979,484.40799 ],[0 ,0, 1] ])
    C, R = combineRC(C0, R1, R2)
    P1 = np.matmul(K,np.hstack((np.eye(3),np.zeros((3,1)))))

    score = [0,0,0,0]
    for i in range(len(R)):
        XPoints = list()
        P2 = np.matmul(K,np.hstack((R[i], np.matmul(R[i],C[i].T))))
        for j in range(len(pts1)):
            D = np.array([pts1[j][0]*P1[2].T - P1[0].T,
                          pts1[j][1]*P1[2].T - P1[1].T,
                          pts2[j][0]*P2[2].T - P2[0].T,
                          pts2[j][1]*P2[2].T - P2[1].T])

            U, s, Vt = np.linalg.svd(D)
            X = Vt[-1,:]
            X = X / X[3]
            XPoints.append(X[:3])
        # print(R[i][2].shape)
        # print(XPoints[0][:].shape)
        # print((XPoints[0][:] - C[i]).reshape((1,3)))
        # print(R[i][2].reshape((3,1)))
        for k in range(len(XPoints)):
            if ( np.matmul(R[i][2].reshape((1,3)) , (XPoints[k][:] - C[i]).reshape((3,1)) )) > 0 :
                score[i] = score[i] + 1

    idx = np.argmax(score)
    R_final = R[idx]
    C_final = C[idx]
    return R_final, C_final

def combineRC(C0, R1, R2):
    R = list()
    C = list()
    C0 = C0.reshape((1,3))
    C.append(C0)
    C.append(-C0)
    C.append(C0)
    C.append(-C0)
    R.append(R1)
    R.append(R1)
    R.append(R2)
    R.append(R2)
    return C, R
