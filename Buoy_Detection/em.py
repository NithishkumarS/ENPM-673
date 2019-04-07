import numpy as np
import pylab as plt
import os, sys
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from scipy.stats import multivariate_normal as mvn

Train = 1
K=4

#Set values here
testfolder = "Test Images"
outfolder = "EM Output"
'''
def compute_gaussian(x,mean, cv):
    print(np.sum((x-mean)@np.linalg.inv(cv)@np.transpose(x-mean),axis =0).shape)
    deter =  np.linalg.det(cv)
    if deter != 0:
        b = (np.exp(-.5*np.sum((x-mean)@np.linalg.inv(cv)@np.transpose(x-mean),axis =1)))
        a = (1/np.sqrt((2*np.pi)**3*deter))* b
    else:
        a = 1
    return a
'''

def EMalgo(xtrain,K,iters, color):
    n,d = xtrain.shape
    mean = xtrain[np.random.choice(n, K, False),:]
    Sigma = [80*np.eye(d)] * K
    for i in range(K):
        Sigma[i]=np.multiply(Sigma[i],np.random.rand(d,d))
        #Sigma[i]=Sigma[i]*Sigma[i].T
    w = [1./K] * K
    z = np.zeros((n, K))
    log_likelihoods = []
    while len(log_likelihoods) < iters:
        for k in range(K):
            tmp = w[k] * mvn.pdf(xtrain, mean[k], Sigma[k], allow_singular=True)
            print(tmp.shape)
            z[:,k] = tmp.reshape((n,))
        log_likelihood = np.sum(np.log(np.sum(z, axis = 1)))
        print('{0} -> {1}'.format(len(log_likelihoods),log_likelihood))
        log_likelihoods.append(log_likelihood)
        z = (z.T / np.sum(z, axis = 1)).T
        N_ks = np.sum(z, axis = 0)
        for k in range(K):
            if (N_ks[k]==0):
                continue
            mean[k] = 1. / N_ks[k] * np.sum(z[:, k] * xtrain.T, axis = 1).T
            x_mean = np.matrix(xtrain - mean[k])
            Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mean.T,  z[:, k]), x_mean))
            w[k] = 1. / n * N_ks[k]
        if len(log_likelihoods) < 2 : continue
        if len(log_likelihoods)>10000 or np.abs(log_likelihood - log_likelihoods[-2]) < 0.0001: break
    plt.plot(log_likelihoods)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
    np.save('weights_' + color, w)
    np.save('sigma_' + color, Sigma)
    np.save('mean_' + color, mean)

colors = ['Red','Green','Yellow']
for i in range(3):
    xtrain = np.load('xtrain_'+ colors[i] +'.npy')
    EMalgo(xtrain,K,10000, colors[i])
