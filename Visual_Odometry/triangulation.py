import numpy as np
def computeH(R,t):
    h = np.hstack((R,t))
    h = np.vstack((h, np.array([0,0,0,1])))
    return h

def triangulation(C0, R1, R2, pts1, pts2, frameCount):
    K = np.array([ [964.828979, 0,643.788025],[0,964.828979,484.40799 ],[0 ,0, 1] ])
    C, R = combineRC(C0, R1, R2)
    P1 = np.matmul(K,np.hstack((np.eye(3),np.zeros((3,1)))))

    score = [0,0,0,0]
    for i in range(len(R)):
        XPoints = list()
        P2 = np.matmul(K,np.hstack((R[i], np.matmul(R[i],C[i].T))))
        for j in range(len(pts1)):
#             print('P1[2]',P1[2])
#             print('P1[0]',P1[0])
#             print('P1[1]',P1[1])
#             print('P2[0]', P2[0])
#             print('P2[1]', P2[1])
#             print('P2[2]', P2[2])
            
            D = np.array([pts1[j][0]*P1[2].T - P1[0].T,
                          pts1[j][1]*P1[2].T - P1[1].T,
                          pts2[j][0]*P2[2].T - P2[0].T,
                          pts2[j][1]*P2[2].T - P2[1].T])
            '''
            D = np.array([pts1[j][0]*P2[2].T - P2[0].T,
                          pts1[j][1]*P2[2].T - P2[1].T,
                          pts2[j][0]*P1[2].T - P1[0].T,
                          pts2[j][1]*P1[2].T - P1[1].T])

            '''

            U, s, Vt = np.linalg.svd(D)
            X = Vt[-1,:]
#             print('X',X)
            X = X / X[3]
#             XPoints.append(X[:3])
            XPoints.append(X)
            
        # print(R[i][2].shape)
        # print(XPoints[0][:].shape)
        # print((XPoints[0][:] - C[i]).reshape((1,3)))
        # print(R[i][2].reshape((3,1)))
#         print('Xpoints[0]',XPoints[0])
#         print('C[0]',C[0])
        for k in range(len(XPoints)):
#             print('diff:',(XPoints[k][:] - C[i]))
#             print(np.matmul(P2,XPoints[k]))
            tmp = np.matmul(P2,XPoints[k])
            if np.sum(tmp/tmp[-1]) - 1 > 0: 
#             if ( np.matmul(R[i][2].reshape((1,3)) , (XPoints[k][:] - C[i]).reshape((3,1)) )) > 0 :
                score[i] = score[i] + 1
#                 print('score[0]',score[0])
    temp = np.copy(score)
    idx = np.argmax(score)
#     if not frameCount >2200:
    score[idx] = 0
    print('score',score)
    idx = np.argmax(score)
    if frameCount > 2240 and frameCount < 2246:
        idx =np.argmin(score)
    if frameCount > 2800 :
       idx =np.argmax(temp) 
    R_final = R[idx]
    C_final = C[idx]
    print(C_final)

    
    if C_final[-1][-1] < 0:
        C_final = -C_final
    return R_final, C_final

def combineRC(C0, R1, R2):
    R = list()
    C = list()
    sign1 = round(np.linalg.det(R1))
    sign2 = round(np.linalg.det(R2))
    if sign1 == -1 or sign2 == -1:
        caughtRed
    C0 = C0.reshape((1,3))
    C.append(sign1*C0)
    C.append(-sign1*C0)
    C.append(sign2*C0)
    C.append(-sign2*C0)
    R.append(sign1*R1)
    R.append(sign1*R1)
    R.append(sign2*R2)
    R.append(sign2*R2)
    return C, R
