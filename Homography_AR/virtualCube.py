import numpy as np
import cv2
import math

def virtualCube(H,frame,cP):
    print(cP)
    K = np.transpose( np.array( [[1406.08415449821,0,0 ], [2.20679787308599, 1417.99930662800,0], [1014.13643417416, 566.347754321696,1]]) )
    print(K)

    #cornerPoints, corner = getCornerPoints(frame)
    #H_intial = homographicTransform(cornerPoints,corner)
    print('homography matrix')
    print(H)
    B = np.matmul(np.linalg.inv(K), np.linalg.inv(H))

    det = np.linalg.det(B)
    print('det:',det)
    if det < 0 :
        B = -B
    print(B)

    term = np.linalg.norm(B[:,0]) +  np.linalg.norm( B[:,1])   #np.linalg.norm(np.matmul(np.linalg.inv(K), H[:,0]) ) +  np.linalg.norm( np.matmul(np.linalg.inv(K), H[:,1]) )
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

    projectionMat1 = projection_matrix(K,np.linalg.inv(H))
    print('projectionMat1')
    print( projectionMat1)
    # h_inv = np.linalg.inv(projectionMat1)
    Xw = np.array([[0,0,-199,1],[199,0,-199,1],[199,199,-199,1],[0,199,-199,1]] )
    Xw = np.transpose(Xw)
    print(Xw)

    Xc = np.matmul(projectionMat,Xw)
    print('Xc:')
    print(Xc)

    Xc[:,0] = Xc[:,0] / Xc[2][0]
    #print(Xc[:,0])
    Xc[:,1] = Xc[:,1] / Xc[2][1]
    Xc[:,2] = Xc[:,2] / Xc[2][2]
    Xc[:,3] = Xc[:,3] / Xc[2][3]
    Xc = Xc.astype(int)

    print('Xc_new')
    print(Xc)
    frame[Xc[1][0], Xc[0][0]] = [255,0,0]
    frame[Xc[1][1], Xc[0][1]] = [255,0,0]
    frame[Xc[1][2], Xc[0][2]] = [255,0,0]
    frame[Xc[1][3], Xc[0][3]] = [255,0,0]

    cv2.line(frame,(Xc[0][0],Xc[1][0] ),( Xc[0][1],Xc[1][1]), (255,0,0),3 )
    cv2.line(frame,(Xc[0][1],Xc[1][1] ),( Xc[0][2],Xc[1][2]), (255,0,0),3 )
    cv2.line(frame,(Xc[0][2],Xc[1][2] ),( Xc[0][3],Xc[1][3]), (255,0,0),3 )
    cv2.line(frame,(Xc[0][3],Xc[1][3] ),( Xc[0][0],Xc[1][0]), (255,0,0),3 )

    cv2.line(frame,(Xc[0][0],Xc[1][0] ),( cP[0][0],cP[0][1]), (255,0,0),3 )
    cv2.line(frame,(Xc[0][1],Xc[1][1] ),( cP[1][0],cP[1][1]), (0,255,0),3 )
    cv2.line(frame,(Xc[0][2],Xc[1][2] ),( cP[2][0],cP[2][1]), (0,255,255),3 )
    cv2.line(frame,(Xc[0][3],Xc[1][3] ),( cP[3][0],cP[3][1]), (0,0,255),3 )


    cv2.imshow('3D cube', frame)



def projection_matrix(camera_parameters, homography):

# Compute rotation along the x and y axis as well as the translation
    homography = homography * (1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)
