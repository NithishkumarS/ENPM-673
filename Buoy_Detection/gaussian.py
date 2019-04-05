import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, pi, exp, erfc
from scipy.stats import norm
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

class Gaussian:
    """
    Model color distribution using 1D Gaussian
    """
    mean_g = 0
    mean_r = 0
    mean_y = 0
    variance_g = 0
    variance_r = 0
    variance_y = 0

    def __init__(self):
        green_buoy_images = []
        red_buoy_images = []
        yellow_buoy_images = []

        # Load Data Set
        for file in os.listdir("DataSet/Green"):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                green_buoy_images.append("DataSet/Green/" + file)
        for file in os.listdir("DataSet/Red"):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                red_buoy_images.append("DataSet/Red/" + file)
        for file in os.listdir("DataSet/Yellow"):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                yellow_buoy_images.append("DataSet/Yellow/" + file)

        self.mean_g, self.variance_g = self.get_mean_gaussians(green_buoy_images)
        self.mean_r, self.variance_r = self.get_mean_gaussians(red_buoy_images)
        self.mean_y, self.variance_y = self.get_mean_gaussians(yellow_buoy_images)

        # print(self.mean_g, self.mean_r, self.mean_y)
        # print(self.variance_g, self.variance_r, self.variance_y)

    def get_histogram(self, img, channel):
        hist = cv2.calcHist([img], [channel], None, [256], [0,256])
        return hist

    def get_mean(self, hist):
        sum = 0
        n = 0
        for i in range(hist.shape[0]):
            sum += (i * hist[i][0])
            n += hist[i][0]
        mean = sum/n
        return mean

    def get_variance(self, hist, mean):
        num = 0
        n = 0
        for i in range(hist.shape[0]):
            num += (hist[i][0])*((i - mean)**2)
            n += hist[i][0]
        variance = num/n
        return variance

    def get_gaussian(self, hist):
        mean = self.get_mean(hist)
        variance = self.get_variance(hist, mean)
        return mean, variance

    def get_mean_gaussians(self, images):
        hists_b = np.zeros((256,1))
        hists_g = np.zeros((256,1))
        hists_r = np.zeros((256,1))

        for image in images:
            img = cv2.imread(image)
            hists_b = np.add(hists_b, self.get_histogram(img, 0))
            hists_g = np.add(hists_g, self.get_histogram(img, 1))
            hists_r = np.add(hists_r, self.get_histogram(img, 2))

        mean_b, variance_b = self.get_gaussian(hists_b)
        mean_g, variance_g = self.get_gaussian(hists_g)
        mean_r, variance_r = self.get_gaussian(hists_r)
        mean = [mean_b, mean_g, mean_r]
        variance = [variance_b, variance_g, variance_r]
        return mean, variance

    def get_pdf(self, x, mean, variance):
        """
        Probability Density Function
        """
        #u = (x-mean)/abs(variance)
        #y = (1/(sqrt(2*pi)*abs(variance)))*exp(-u*u/2)
        # var = float(variance)
        # denom = (2*pi*var)**0.5
        # num = exp(-(float(x)-float(mean))**2/(2*var))
        # return num/denom

        # TOO SLOW ---> Need to replace
        # https://stackoverflow.com/questions/809362/how-to-calculate-cumulative-normal-distribution-in-python
        # cdf = norm.cdf(x, mean, sqrt(variance))
        # return cdf
        t = x-mean;
        y = 0.5*erfc(-t/(variance*sqrt(2.0)));
        if y>1.0:
            y = 1.0;
        return y
        #return y

    def get_thresholded_pdf(self, frame, p1):
        # q = np.multiply(np.multiply(np.multiply(frame, p1>thresh1), p2>thresh2), p3>thresh3)
        # q1 = (p1>0.00804).astype(int)
        # q2 = (p1<0.0104).astype(int)

        # q3 = (p2>0.011).astype(int)
        # q4 = (p2<0.012).astype(int)
        #
        # q5 = (p3>0.008).astype(int)
        # q6 = (p3<0.0091).astype(int)
        # q = np.multiply(q1, np.multiply(q2, np.multiply(np.multiply(q3, np.multiply(q4, q5)), q6))).astype(np.uint8)
        # q = np.multiply(frame, p1)
        frame[p1 == True] = 255
        frame[p1 ==False] =0
        return frame

    def skel(self, img):
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)

        ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False

        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True
        return skel

    def detect_buoys(self, frame):
        frame_b = frame[:,:,0]
        frame_g = frame[:,:,1]
        frame_r = frame[:,:,2]

        # Function mapping
        # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array

        # For Green Buoy - RBG PDF
        map_fn_g_b = lambda x:self.get_pdf(x, self.mean_g[0], self.variance_g[0])
        fn_g_b = np.vectorize(map_fn_g_b)
        pdf_g_b = fn_g_b(frame_b)


        map_fn_g_g = lambda x:self.get_pdf(x, self.mean_g[1], self.variance_g[1])
        fn_g_g = np.vectorize(map_fn_g_g)
        pdf_g_g = fn_g_g(frame_g)

        map_fn_g_r = lambda x:self.get_pdf(x, self.mean_g[2], self.variance_g[2])
        fn_g_r = np.vectorize(map_fn_g_r)
        pdf_g_r = fn_g_r(frame_r)

        # For Red Buoy - RBG PDF
        map_fn_r_b = lambda x:self.get_pdf(x, self.mean_r[0], self.variance_r[0])
        fn_r_b = np.vectorize(map_fn_r_b)
        pdf_r_b = fn_r_b(frame_b)

        map_fn_r_g = lambda x:self.get_pdf(x, self.mean_r[1], self.variance_r[1])
        fn_r_g = np.vectorize(map_fn_r_g)
        pdf_r_g = fn_r_g(frame_g)

        map_fn_r_r = lambda x:self.get_pdf(x, self.mean_r[2], self.variance_r[2])
        fn_r_r = np.vectorize(map_fn_g_r)
        pdf_r_r = fn_r_r(frame_r)

        # plt.figure(1)
        # x = np.linspace(0,256,256)
        # plt.plot(x,norm.pdf(x, self.mean_r[2], self.variance_r[2]),"r-")
        # plt.hist(frame_r.ravel(),256,(0,256))
        # plt.show()

        # For Yellow Buoy - RBG PDF
        map_fn_y_b = lambda x:self.get_pdf(x, self.mean_y[0], self.variance_y[0])
        fn_y_b = np.vectorize(map_fn_y_b)
        pdf_y_b = fn_y_b(frame_b)

        map_fn_y_g = lambda x:self.get_pdf(x, self.mean_y[1], self.variance_y[1])
        fn_y_g = np.vectorize(map_fn_y_g)
        pdf_y_g = fn_y_g(frame_g)

        map_fn_y_r = lambda x:self.get_pdf(x, self.mean_y[2], self.variance_y[2])
        fn_y_r = np.vectorize(map_fn_y_r)
        pdf_y_r = fn_y_r(frame_r)

        # Thresold by Tial and Error

        print('R max ',np.amax(pdf_r_r))
        print('G max ',np.amax(pdf_r_g))
        print('B max ',np.amax(pdf_r_b))

        # Max PDF G G 0.00063265
        _frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        print('Noise')
        print(pdf_r_r[227,150])
        print(pdf_r_g[227,150])
        print(pdf_r_b[227,150])

        print('Buoy')
        print(pdf_r_r[195,342])
        print(pdf_r_g[195,342])
        print(pdf_r_b[195,342])
        print((pdf_r_r > 0.00027704).shape)
        print(frame_r.shape)
        #_frame_r = cv2.GaussianBlur(frame_r,(5,5),0)
        _frame_r = self.get_thresholded_pdf(frame_r, pdf_r_r > .99*np.amax(pdf_r_r))
        #_frame_r = cv2.GaussianBlur(_frame_r,(5,5),0)
        kernel = np.ones((9,9),np.uint8)
        _frame_r = cv2.morphologyEx(_frame_r, cv2.MORPH_OPEN, kernel)
        #_frame_r = cv2.medianBlur(_frame_r,3)
        #kernel = np.ones((7,7),np.uint8)
        #_frame_r = cv2.erode(_frame_r,kernel,iterations = 1)
        kernel = np.ones((9,9),np.uint8)
        _frame_r = cv2.dilate(_frame_r,kernel,iterations = 1)
        #temp = self.skel(_frame_r)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
        #_frame_r = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        # R 0.0105136
        # G 0.01210605
        # B 0.009138

        # Max PDF G B 0.00048706450619803384
        # _frame_b = self.get_thresholded_pdf(frame_b, pdf_g_b, 0.00048)
        # Max PDF G R 0.0002954202761348719
        # _frame_r = self.get_thresholded_pdf(frame_r, pdf_g_r, 0.00029)
        # print(p)
        # print(np.array([_frame_b, _frame_g, _frame_r]))
        '''
        std_r_r = sqrt(self.variance_r[2])
        x_r = np.linspace(0,255,256)
        #x_r = np.linspace(self.mean_r[2] - 3*std_r_r,self.mean_r[2] + 3*std_r_r, 480)
        #plt.plot(self.get_histogram(frame, 2),'r')
        histr_r = self.get_histogram(frame, 2)
        mr_r, vr_r = self.get_gaussian(histr_r)
        pdf = np.vectorize(self.get_pdf)
        pdfr_r = pdf(histr_r, mr_r, vr_r)
        print(np.amax(pdfr_r))
        plt.plot(x_r, pdfr_r,'r')
        _frame_r = self.get_thresholded_pdf(frame_r,pdfr_r)
        '''
        #plt.plot(self.get_histogram(frame, 1),'g')
        #plt.plot(self.get_histogram(frame, 0),'b')
        #plt.show()

        return _frame_r
