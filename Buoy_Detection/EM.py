import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import math


def compute_gaussian(x,mean, cv):
    
    deter =  np.linalg.det(cv)
   # b = (np.exp( -.5*np.matmul(np.matmul((x-mean),np.linalg.inv(cv)),np.transpose(x-mean))  ))
    b = (np.exp( -.5*(x-mean)*np.linalg.inv(cv)*np.transpose(x-mean)  ))
    
   # print(np.matmul(np.matmul((x-mean),np.linalg.inv(cv)),np.transpose(x-mean)))
    a = (1/np.sqrt((2*np.pi)**3*deter))* b
   # if a == 0:
    #    a =0.0001
    return a

def compute_latent(weight, data_x, mean, cv ):
    return weight* compute_gaussian(data_x, mean, cv)

def compute_EM(no_of_clusters , data):
    l = 1#len(data[0])
    mean = []    
#     mean = data[np.random.choice(len(data), no_of_clusters, replace=False), :]

    for i in np.random.choice(len(data), no_of_clusters, replace=False):
        mean.append(data[i])
    cv = [np.identity(l)]* no_of_clusters
    print('cv                 row:',len(cv))
    print('cv                 col:',len(cv[0]))
    print(cv)

    weight =  [ 1./no_of_clusters for i in range(no_of_clusters)]  
    count = 0
    denom = 0
    past = 0
    temp = 1
    while count < 1000:
            
        A = np.zeros((len(weight),l))#len(data[0])))
        B = np.zeros(len(weight))
        C = np.zeros_like(cv)
           
        #E Method
        '''for i in range(0,len(weight)):
            print('mean:',i,' ',mean[i])
            #latent = weight[i] * compute_gaussian(data, mean[i], cv[i]) / denom
        '''
    
        
        for i in range(len(weight)):
            for n in range(len(data)):
                data_x = data[n]#.reshape((1,3))
              #  print('size:::::::::::::::',data_x.shape)
        
                denom = 0
                for j in range(0,len(weight)):
             #       print('i:',i,' n',n,' j:',j)
                    denom = denom + compute_latent(weight[j],data_x, mean[j], cv[j])
                print(weight[i],' ', data_x,' ',  mean[i],' ', cv[i])
                latent_variable = compute_latent(weight[i], data_x, mean[i], cv[i])/denom
                print('latent Variable',latent_variable)
                A[i] = A[i] + (latent_variable)*data_x
                
                B[i] = B[i] + latent_variable
                if B[i] ==0:
                    print('A::',A[i])
                    temp = 0
                    break
                #print('A:',B[i])
               # C[i] = C[i] + (latent_variable)*np.matmul(np.transpose((data_x - mean[i])),(data_x - mean[i]) )
                C[i] = C[i] + (latent_variable)*np.transpose((data_x - mean[i]))*(data_x - mean[i]) 
                #print(np.matmul(np.transpose((data_x - mean[i])),(data_x - mean[i]) ))
                print('iteration: ',n)
           # print('A:',A[i])
            #print('B:',B[i])
            #print('C:',
            
            print('temp', temp)
            #if temp == 0:
             #  break
            mean[i] = (1/B[i])*A[i]
            cv[i] = C[i]/B[i]
            weight[i] = B[i]/(len(data))
        #print('weight',weight)
        likelihood = 0
        #if temp == 0:
         #   break
        for n in range(len(data)):
            data_x = data[n]
            latent =0 
            for i in range(len(weight)):
              latent = latent + compute_latent(weight[i],data_x, mean[i], cv[i])        
            likelihood = likelihood + np.log(latent)  
        print past - likelihood
        if abs(past - likelihood) < .001:
            print('mean:',mean)
            print('covariance:', cv)
            print('weights',weight)
            break
        
        past = likelihood
        count = count +1
        print('mean:',mean)
        print('covariance:', cv)
        print('weights',weight)
        print('count: ',count)
        
        mu = mean[2]
        variance = cv
        sigma = np.zeros(3)
        sigma[0] = math.sqrt(variance[0])
        sigma[1] = math.sqrt(variance[1])
        sigma[2] = math.sqrt(variance[2])
        x = np.linspace(int(mean[2] - round(3*sigma[2])), int(mean[0] + round(3*sigma[0])), 100)
        plt.plot(x, weight[0]*stats.norm.pdf(x, mean[0], sigma[0])+weight[1]*stats.norm.pdf(x,mean[1], sigma[1])+weight[2]*stats.norm.pdf(x, mean[2], sigma[2]))
        
    plt.show()
    

def main(): # mean , variance
    np.random.seed(0)
    '''
    m1 =  [0,0,0]
    cv1 = [[1,0,0],[0,1,0],[0,0,1]]
    m2 =  [5,5,5]
    cv2 = [[1,0,1],[0,1,1],[1,0,1]]
    m3 =  [2,2,2]
    cv3 = [[1,.5,0],[2,1,0],[1,0,1]]
    s1 = np.random.multivariate_normal(m1,cv1, 50)
    s2 = np.random.multivariate_normal(m2,cv2, 50)
    s3 = np.random.multivariate_normal(m3,cv3, 50)
    '''
    mean1 = 0 
    mean2 = 3
    mean3 = 6
    cv1 = 2
    cv2 = .5
    cv3 = 3
    s1 = np.random.normal(mean1, cv1, 100)
    s2 = np.random.normal(mean2, cv2, 100)
    s3 = np.random.normal(mean3, cv3, 100)
    print('s1',type(s1))
    
    data = []
    
    for i in s1:
        data.append(i)
    for i in s2:
        data.append(i)
    for i in s3:
        data.append(i)
    print('Fucntion call')
    no_of_clusters = 3
    compute_EM(no_of_clusters, data)
    

    
if __name__ == "__main__":
    main()



'''
print(s3)


print ( abs(0 - np.mean(s1)) < 0.01)
import matplotlib.pyplot as plt

plt.scatter(s1,[i for i in range(0,100)])
plt.show()
'''

