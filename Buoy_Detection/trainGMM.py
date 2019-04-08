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


thresh = 0.0001

def compute_gaussian(x,mean, cv):
    print(np.sum((x-mean)@np.linalg.inv(cv)@np.transpose(x-mean),axis =0).shape)
    deter =  np.linalg.det(cv)
    if deter != 0:
        b = (np.exp(-.5*np.sum((x-mean)@np.linalg.inv(cv)@np.transpose(x-mean),axis =1)))
        a = (1/np.sqrt((2*np.pi)**3*deter))* b
    else:
        a = 1
    return a


def compute_EM(channel,no_of_gaussians,iters, color):
    n,d = channel.shape
    mean = channel[np.random.choice(n, no_of_gaussians, False),:]
    cv = [80*np.eye(d)] * no_of_gaussians
    for i in range(no_of_gaussians):
        cv[i]=np.multiply(cv[i],np.random.rand(d,d))
        #cv[i]=cv[i]*cv[i].T
    weight = [1./no_of_gaussians] * no_of_gaussians
    latent_variable = np.zeros((n, no_of_gaussians))
    log_likelihoods = []
    while len(log_likelihoods) < iters:
        print('Iteration number :',len(log_likelihoods))
        for k in range(no_of_gaussians):
            calc = weight[k] * mvn.pdf(channel, mean[k], cv[k], allow_singular=True)
            latent_variable[:,k] = calc.reshape((n,))
        log_likelihood = np.sum(np.log(np.sum(latent_variable, axis = 1)))
        log_likelihoods.append(log_likelihood)
        latent_variable = (latent_variable.T / np.sum(latent_variable, axis = 1)).T
        latent_sum = np.sum(latent_variable, axis = 0)
        for k in range(no_of_gaussians):
            if (latent_sum[k]==0):
                continue
            mean[k] = 1. / latent_sum[k] * np.sum(latent_variable[:, k] * channel.T, axis = 1).T
            x_mean = np.matrix(channel - mean[k])
            cv[k] = np.array(1 / latent_sum[k] * np.dot(np.multiply(x_mean.T,  latent_variable[:, k]), x_mean))
            weight[k] = 1. / n * latent_sum[k]
        if len(log_likelihoods) < 2 :
            continue
        if np.abs(log_likelihood - log_likelihoods[-2]) < thresh :
            break
    np.save('parameters/weights_' + color, weight)
    np.save('parameters/cv_' + color, cv)
    np.save('parameters/mean_' + color, mean)

no_of_gaussians=4
colors = ['Red','Green','Yellow']
for i in range(3):
    channel = np.load('Input/channel_'+ colors[i] +'.npy')
    compute_EM(channel,no_of_gaussians,10000, colors[i])
