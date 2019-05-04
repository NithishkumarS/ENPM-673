import numpy as np

def projection(R , C , X ,K):
    
    
    tmp = np.concatenate((R.T , np.matmul(-R.T,C.T).reshape((1,3)).T ),axis = 1)
    return np.matmul( np.matmul(K,np.matmul(tmp, X)))

def triangulation(R, T, K, P1):
 
    X = []   
    for i in range(len(P1)):
        x_new = np.array([P1[i][0], P1[i][0], 1])
        X.append(x_new)
    X = np.array(X)
    X3d = projection(R,T,X.T, K)
    
    X3d[:,i]