import numpy as np
import math
import cv2
def houghTransform(binaryImage, frame):
    
    Image = frame
    #Image = cv2.imread('test.jpeg')
   # im_gray = cv2.imread('test.jpeg', cv2.IMREAD_GRAYSCALE)
    
   # thresh = 127
    #binaryImage = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    #print binaryImage
    #frame = binaryImage
    
    
    x_min = 300  #25 
    x_max = 1100 #215
    y_min = 400  #69
    y_max = 650 # 145
    
    max_len = int(np.ceil(math.sqrt(x_max*x_max + y_max*y_max) )) 
   # print(max_len)
    lut = np.zeros((2*max_len, 180))
   # print('lut size:',lut.shape)
    
    for i in range(y_min, y_max) :
        for j in range( x_min, x_max) :
           # print('i',i )
           # print('j',j)
            if binaryImage[i,j] == 255 :
                for theta in range(0,180):
    #                print('theta',theta)
                    d =int(round( j*(math.cos(theta*math.pi/180)) + i*(math.sin(theta*math.pi/180)) ))
                    #print(d,theta)
                    lut[d,theta] +=1
                    
    cv2.imshow('lut',lut.T)
    status = 0
    d_thresold = 0
    th_thresold = 0
    for i in range(0,50):
        loc1 = np.unravel_index(lut.argmax(), lut.shape)
        lut[loc1] = 0
        loc2 = np.unravel_index(lut.argmax(), lut.shape)
      
        d = loc1[0]
        th = loc1[1]
        if i == 0:
            d_thresold = d
            th_thresold = th
        elif  d_thresold - d < 5:
            d =  d_thresold
            th = th_thresold
            status = 1 
            
        print ('znncknncccccccccccccccccccccccccccccccccccccccccc', d)
        
        if th != 0 and status ==1:
            x1 = x_max
            y1 = int((d - math.cos(math.radians(th))*x1) / math.sin(math.radians(th))) 
            x2 = x_min
            y2 = int((d - math.cos(math.radians(th))*x2) / math.sin(math.radians(th)))
            print(x1,y1,x2,y2)
            cv2.line(Image, (x1, y1), (x2, y2), (0,255,0))
            status = 0    
    '''
    # d = loc2[0]
    #th = loc2[1]
    
    print ('znncknncccccccccccccccccccccccccccccccccccccccccc', d)
    if th != 0 :    
        x1 = x_max
        y1 = int((d - math.cos(math.radians(th))*x1) / math.sin(math.radians(th)))
        x2 = x_min
        y2 = int((d - math.cos(math.radians(th))*x2) / math.sin(math.radians(th))) 
#        print(x1,y1,x2,y2)
        cv2.line(frame, (x1, y1), (x2, y2), (0,0,255))
    '''
    cv2.imshow('op',Image)