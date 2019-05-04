import numpy as np

def projection(R , T , X ,K):
    tmp = np.concatenate((np.eye(3),T.T),axis = 1)
    
    return np.matmul((np.matmul(K,np.matmul(R,tmp))) , X)

def triangulation(R, T, K, P1):
 
    X = []   
    for i in range(len(P1)):
        x_new = np.array([P1[i][0], P1[i][0], 1])
        X.append(x_new)
    X = np.array(X)
    X3d = projection(R,T,X.T, K)
    
    X3d[:,i]