import os
import sys
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import numpy as np
from math import sqrt, pi, exp, erfc
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
    green_buoy_images = []
    red_buoy_images = []
    yellow_buoy_images = []

    def __init__(self):

        # Load Data Set
        for file in os.listdir("DataSet/Green"):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                self.green_buoy_images.append("DataSet/Green/" + file)
        for file in os.listdir("DataSet/Red"):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                self.red_buoy_images.append("DataSet/Red/" + file)
        for file in os.listdir("DataSet/Yellow"):
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                self.yellow_buoy_images.append("DataSet/Yellow/" + file)

        self.mean_g, self.variance_g = self.get_mean_gaussians(self.green_buoy_images)
        self.mean_r, self.variance_r = self.get_mean_gaussians(self.red_buoy_images)
        self.mean_y, self.variance_y = self.get_mean_gaussians(self.yellow_buoy_images)

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
        t = x-mean;
        y = 0.5*erfc(-t/(variance*sqrt(2.0)));
        if y>1.0:
            y = 1.0;
        return y

    def get_thresholded_pdf(self, frame, p1):
        frame[p1 == True] = 255
        frame[p1 == False] = 0
        return frame

    def draw_buoy_contour(self, original_frame, reference_frame, color):
        contours, hier = cv2.findContours(reference_frame.copy(), 1, 2)
        radius_r = []
        if contours:
            for c in contours:
                point,radius = cv2.minEnclosingCircle(c)
                radius_r.append(int(radius))

            max_r = np.argmax(radius_r)
            cnt = contours[max_r]
            moments = [cv2.moments(cnt)]

            centroids = [(int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) for M in moments]
            for c in centroids:
                cv2.circle(original_frame, c, radius_r[max_r], color, thickness=2)
            print(centroids)
        return original_frame

    def plot(self):
        x_r = np.linspace(0,255,256)
        img = cv2.imread(self.yellow_buoy_images[0])

        map = np.vectorize(self.get_pdf)

        plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
        plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
        y_r = map(x_r, self.mean_g[0], self.variance_g[0])
        #plt.show()
        plt.savefig('plots/hists_1D_yBouy.png')
        plt.close()


        plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
        y_r = map(x_r, self.mean_r[0], self.variance_r[0])
        #plt.show()
        plt.savefig('plots/hists_1D_rBouy.png')
        plt.close()

        plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
        y_r = map(x_r, self.mean_g[0], self.variance_g[0])
        #plt.show()
        plt.savefig('plots/hists_1D_gBouy.png')
        plt.close()


        #plt.close()

        return 0

    def detect_buoys(self, original_frame, i):
        frame = original_frame.copy()
        frame_b = frame[:,:,0]
        frame_g = frame[:,:,1]
        frame_r = frame[:,:,2]

        # For Green Buoy - RBG PDF
        # map_fn_g_b = lambda x:self.get_pdf(x, self.mean_g[0], self.variance_g[0])
        # fn_g_b = np.vectorize(map_fn_g_b)
        # pdf_g_b = fn_g_b(frame_b)

        map_fn_g_g = lambda x:self.get_pdf(x, self.mean_g[1], self.variance_g[1])
        fn_g_g = np.vectorize(map_fn_g_g)
        pdf_g_g = fn_g_g(frame_g)

        map_fn_g_r = lambda x:self.get_pdf(x, self.mean_g[2], self.variance_g[2])
        fn_g_r = np.vectorize(map_fn_g_r)
        pdf_g_r = fn_g_r(frame_r)

        # For Red Buoy - RBG PDF
        # map_fn_r_b = lambda x:self.get_pdf(x, self.mean_r[0], self.variance_r[0])
        # fn_r_b = np.vectorize(map_fn_r_b)
        # pdf_r_b = fn_r_b(frame_b)
        #
        # map_fn_r_g = lambda x:self.get_pdf(x, self.mean_r[1], self.variance_r[1])
        # fn_r_g = np.vectorize(map_fn_r_g)
        # pdf_r_g = fn_r_g(frame_g)
        #

        map_fn_r_r = lambda x:self.get_pdf(x, self.mean_r[2], self.variance_r[2])
        fn_r_r = np.vectorize(map_fn_g_r)
        pdf_r_r = fn_r_r(frame_r)

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
        # Red Buoy

        kernel = np.ones((9,9),np.uint8)
        _frame_r = self.get_thresholded_pdf(frame_r, pdf_r_r > .99*np.amax(pdf_r_r))
        # cv2.imwrite('plots/Red_bouy_1D_initial.jpg', _frame_r)
        _frame_r = cv2.morphologyEx(_frame_r, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite('plots/Red_bouy_1D_morph.jpg', _frame_r)
        _frame_r = cv2.dilate(_frame_r,kernel,iterations = 1)
        # cv2.imshow('R frame ',_frame_r)
        # cv2.waitKey(0)
        # cv2.imwrite('plots/Red_bouy_1D_dilated.jpg', _frame_r)
        original_frame = self.draw_buoy_contour(original_frame, _frame_r, (0, 0, 255))
        # cv2.imwrite('plots/Red_bouy_1D.jpg', original_frame)


        # Green Buoy
        if (i < 40):
            kernel = np.ones((7,7),np.uint8)
            _frame_gg = self.get_thresholded_pdf(frame_g, pdf_g_g > .97*np.amax(pdf_g_g))
            _frame_gr = self.get_thresholded_pdf(frame_r, pdf_g_r < .97*np.amax(pdf_g_r))
            _frame_gg = cv2.erode(_frame_gg, kernel)
            _frame_g = np.bitwise_and(_frame_gg, _frame_gr)
            _frame_g = cv2.dilate(_frame_g,kernel,iterations = 1)
            original_frame = self.draw_buoy_contour(original_frame, _frame_g, (0, 255, 0))

        # Yellow Buoy

        kernel = np.ones((7,7),np.uint8)
        _frame_yr = self.get_thresholded_pdf(frame_r, pdf_y_r > .75*np.amax(pdf_y_r))
        _frame_yg = self.get_thresholded_pdf(frame_g, pdf_y_g > .95*np.amax(pdf_y_g))
        _frame_yb = self.get_thresholded_pdf(frame_b, pdf_y_b > .95*np.amax(pdf_y_b))
        _frame_yr = cv2.erode(_frame_yr, kernel)
        _frame_yg = cv2.erode(_frame_yg, kernel)
        _frame_yb = np.bitwise_not(_frame_yb)
        _frame_y = np.bitwise_or(_frame_yr, _frame_yg)
        _frame_y = np.bitwise_and(_frame_y, _frame_yb)
        kernel = np.ones((9,9),np.uint8)
        _frame_y = cv2.dilate(_frame_y,kernel,iterations = 1)
        _frame_y = np.bitwise_and(_frame_y, np.bitwise_not(_frame_r))
        original_frame = self.draw_buoy_contour(original_frame, _frame_y, (0, 255, 255))

        return original_frame



if __name__ == '__main__':
    guass = Gaussian()
    guass.plot()
