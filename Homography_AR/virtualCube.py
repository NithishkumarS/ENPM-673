import numpy as np
import cv2


def virtualCube(H,frame):
        
    K = np.transpose( np.array( [[1406.08415449821,0,0 ], [2.20679787308599, 1417.99930662800,0], [1014.13643417416, 566.347754321696,1]]) )
    print(K)
    
    #cornerPoints, corner = getCornerPoints(frame)
    #H_intial = homographicTransform(cornerPoints,corner)
    
    print('homography matrix')
    print(H)
    B = np.matmul(np.linalg.inv(K), H)
    det = np.linalg.det(B)
    print('det:',det)
    if det < 0 :
        B = -B
    print(B)
    
    term =  np.linalg.norm(np.matmul(np.linalg.inv(K), H[:,0]) ) +  np.linalg.norm( np.matmul(np.linalg.inv(K), H[:,1]) )
    lambda1 =  2/term
    
    print('Lambda:',lambda1)
    r1 = lambda1 * B[:,0]
    r2 = lambda1 * B[:,1]
    r3 = np.cross(r1,r2)
    t = lambda1 * B[:,2]
    print('r1')
    print(r1)
    rotationMat = np.array([r1,r2, r3, t]).T
    print(rotationMat)
    projectionMat = np.matmul(K,rotationMat)
    print('projectionMat')
    print( projectionMat)
    
    Xw = np.array([[0,0,-199,1],[199,0,-199,1],[199,199,-199,1],[0,199,-199,1]] )
    Xw = np.transpose(Xw)
        
    Xc = np.matmul(projectionMat,Xw)
    print('Xc:')
    print(Xc.shape)
    
    Xc[:,0] = Xc[:,0] / Xc[2][0]
    #print(Xc[:,0])
    Xc[:,1] = Xc[:,1] / Xc[2][1]
    Xc[:,2] = Xc[:,2] / Xc[2][2]
    Xc[:,3] = Xc[:,3] / Xc[2][3]
    Xc = Xc.astype(int)
    
    print('Xc_new')
    print(Xc)
    frame[Xc[0][0], Xc[1][0]] = [255,0,0]
    frame[Xc[0][1], Xc[1][1]] = [255,0,0]
    frame[Xc[0][2], Xc[1][2]] = [255,0,0]
    frame[Xc[0][3], Xc[1][3]] = [255,0,0]

    #cv2.rectangle(frame,(Xc[0][0],Xc[1][0]),(Xc[0][2], Xc[1][2]),(255,0,0))
    #cv2.rectangle(frame,(Xc[0][2],Xc[1][2]),(Xw[0][1], Xw[1][1]),(255,0,0),2)
   # cv2.rectangle(frame,(Xc[0][0],Xc[1][0]),(Xw[0][1], Xw[1][1]),(0,0,255))
    #cv2.rectangle(frame,(Xc[0][3],Xc[1][3]),(Xw[0][0], Xw[1][0]),(0,0,255))
    #cv2.rectangle(frame,(Xc[0][3],Xc[1][3]),(Xw[0][2], Xw[1][2]),(0,255,0))
    
    cv2.imshow('3D cube', frame) 
    