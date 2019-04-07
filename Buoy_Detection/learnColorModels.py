import numpy as np
from docutils.nodes import row
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import math
from gaussian import Gaussian

def compute_gaussian(x,mean, cv):
    mean = mean.reshape(1,3)
    print('mean    :', mean)
    deter =  np.linalg.det(cv)
   # b = (np.exp( -.5*(x-mean)*np.linalg.inv(cv)*np.transpose(x-mean)  ))
    '''
    print('mean sha', mean.shape)
    print('x',x)
    print('diff',x[0,0] - mean[0,0])
    print('func:', x - mean)
    '''
    b = (np.exp( -.5*np.matmul(np.matmul((x-mean),np.linalg.inv(cv)),np.transpose(x-mean))  ))
#     print('matmul: ',np.matmul(np.matmul((x-mean),np.linalg.inv(cv)),np.transpose(x-mean)))
    a = (1/np.sqrt((2*np.pi)**3*deter))* b

    
   # if a == 0:
    #    a =0.0001
    return a

def compute_latent(weight, data_x, mean, cv ): 
    b = weight* compute_gaussian(data_x, mean, cv)
    return float(b)
def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
    
        vals, vecs = eigsorted(cov)
        print(type(vecs))
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(abs(vals))
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
        ax.add_artist(ellip)
        return ellip 
def show(X, mu, cov):

    plt.cla()
    K = len(mu) # number of clusters
    colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
    plt.plot(X.T[0], X.T[1], 'm*')
    for k in range(K):
      plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  

    
    fig = plt.figure(figsize = (13, 6))
    fig.add_subplot(121)
    show(X, params.mu, params.Sigma)
    fig.add_subplot(122)
    plt.plot(np.array(params.log_likelihoods))
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()

def compute_EM(no_of_clusters , data):
    debug = True
    #debug = False
    print(data.shape)
    ch,col = data.shape
    print('Dimensions',ch,' ',col)
    mean = []    
    data = data.astype(float)
    for i in np.random.choice(len(data[0]), no_of_clusters, replace=False):
        mean.append(data[:,i])
    cv = [np.identity(ch)]* no_of_clusters
    weight =  [ 1./no_of_clusters for i in range(no_of_clusters)]  

    count = 0
    denom = 0
    past = 0
    temp = 1
    while count < 10:
            
        A = np.zeros((len(weight),ch))#len(data[0])))
        B = np.zeros(len(weight))
        C = np.zeros_like(cv)
           
        #E Method       
        for i in range(0,len(weight)):
            for n in range(0,col):
                data_x = data[:,n].reshape((1,3))

               # if debug == True:
                  #  print('size:::::::::::::::',data_x.shape)
        
                denom = 0
                for j in range(0,len(weight)):
                    print('denom')
                    denom = denom + compute_latent(weight[j],data_x, mean[j], cv[j])
                   # print denom
                if denom == 0:
                    continue
                #print(weight[i],' ', data_x,' ',  mean[i],' ', cv[i])
                
                latent_variable = compute_latent(weight[i], data_x, mean[i], cv[i])/denom
                A[:,i] = A[:,i] + (latent_variable)*data_x
                B[i] = B[i] + latent_variable
                if B[i] ==0:
                    temp = 0
                    continue
               # print(np.transpose((data_x - mean[i]))*(data_x - mean[i]))
 
                C[i] = C[i] + (latent_variable)*np.matmul(np.transpose((data_x - mean[i])),(data_x - mean[i])) 
                
               # C[i] = C[i] + (latent_variable)*np.matmul(np.transpose((data_x - mean[i])),(data_x - mean[i]) )
                print('iteration: ',n)
           # print('A:',A[i])
            #print('B:',B[i])
            #print('C:',
            mean[i] = (1/B[i])*A[:,i]
            cv[i] = C[i]/B[i]
            weight[i] = B[i]/(col)
        print('weight',weight)
        print('mean:', mean)
        print('cv',cv)
        #break
        likelihood = 0
        #if temp == 0:
         #   break
        for n in range(col):
            data_x = data[:,n]
            latent =0 
            for i in range(len(weight)):
              latent = latent + compute_latent(weight[i],data_x, mean[i], cv[i])        
            likelihood = likelihood + np.log(latent)  
    #    print past - likelihood
        if abs(past - likelihood) < .000001:
            print('mean:',mean)
            print('covariance:', cv)
            print('weights',weight)
            break
        
        past = likelihood
        count = count +1
        if debug == True:
            print('mean:',mean)
            print('covariance:', cv)
            print('weights',weight)
            print('count: ',count)
        show(data, mean, cv)
        '''
        mu = mean[2]
        variance = cv
        sigma = np.zeros(3)
        sigma[0] = math.sqrt(variance[0])
        sigma[1] = math.sqrt(variance[1])
        sigma[2] = math.sqrt(variance[2])
        x = np.linspace(int(mean[2] - round(3*sigma[2])), int(mean[0] + round(3*sigma[0])), 100)
        plt.plot(x, weight[0]*stats.norm.pdf(x, mean[0], sigma[0])+weight[1]*stats.norm.pdf(x,mean[1], sigma[1])+weight[2]*stats.norm.pdf(x, mean[2], sigma[2]))
        
    plt.show()
        '''

def main(): # mean , variance
    np.random.seed(0)
    gaussian = Gaussian()
    green, red, yellow = gaussian.getBuoys()
    cou = 1
    time = 1
    data_r = np.zeros((1,19))
    data_g = np.zeros((1,19))
    data_b = np.zeros((1,19))
    no_of_clusters = 3
    for image in green:
            if(cou < 3):
                img = cv2.imread(image)
                redChannel = img[:,:,0]
                #print('red: ',redChannel.shape)
                greenChannel = img[:,:,1]
                blueChannel = img[:,:,2]
                for (row1,row2,row3) in zip(redChannel,greenChannel,blueChannel):
                    if time ==1:
                       data_r = row1
                       data_g = row2
                       data_b = row3
                       time = 3
                    else:
                        data_r = np.append(data_r,row1,axis=0)
                        data_g = np.append(data_g,row2,axis=0)
                        data_b = np.append(data_b,row3,axis=0)
                cou = 2
    data_r = np.array(data_r).flatten()
    data_b = np.array(data_b).flatten()
    data_g = np.array(data_g).flatten()
    b = np.vstack((data_g,data_b))
    data = np.vstack((data_r,b))
    print(data[0].shape)
    compute_EM(no_of_clusters, data)
    
    
if __name__ == "__main__":
    main()