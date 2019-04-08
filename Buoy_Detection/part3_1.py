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
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold = sys.maxsize)

def plot3D(yellow_buoy_images, green_buoy_images, red_buoy_images):
    img = cv2.imread(yellow_buoy_images[0])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #for x in range()
    ax.scatter(img[:,:,0].ravel(), img[:,:,1].ravel(), img[:,:,2].ravel(),'r')
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.savefig('plots/scatter_yellow.png')
    plt.close()

    img = cv2.imread(green_buoy_images[0])
    i,j,k = img.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #for x in range()
    ax.scatter(img[:,:,0].ravel(), img[:,:,1].ravel(), img[:,:,2].ravel(),'r')
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.savefig('plots/scatter_green.png')
    plt.close()

    img = cv2.imread(red_buoy_images[0])
    i,j,k = img.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #for x in range()
    ax.scatter(img[:,:,0].ravel(), img[:,:,1].ravel(), img[:,:,2].ravel(),'r')
    ax.set_xlabel('Blue')
    ax.set_ylabel('Green')
    ax.set_zlabel('Red')
    plt.savefig('plots/scatter_red.png')
    plt.close()
    return

def findHist(yellow_buoy_images, green_buoy_images, red_buoy_images):
    x_r = np.linspace(0,255,256)

    color_channels = ['b','g','r']
    img = cv2.imread(yellow_buoy_images[0])

    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "blue")
    plt.savefig('plots/hists_Ybouy_b_ch.png')
    plt.close()
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
    plt.savefig('plots/hists_Ybouy_g_ch.png')
    plt.close()
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
    plt.savefig('plots/hists_Ybouy_r_ch.png')
    plt.close()

    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "blue")
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
    plt.savefig('plots/hists_Ybouy_total.png')
    plt.close()

    img = cv2.imread(green_buoy_images[0])

    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "blue")
    plt.savefig('plots/hists_Gbouy_b_ch.png')
    plt.close()
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
    plt.savefig('plots/hists_Gbouy_g_ch.png')
    plt.close()
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
    plt.savefig('plots/hists_Gbouy_r_ch.png')
    plt.close()

    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "blue")
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
    plt.savefig('plots/hists_Gbouy_total.png')
    plt.close()

    img = cv2.imread(red_buoy_images[0])
    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "blue")
    plt.savefig('plots/hists_Rbouy_b_ch.png')
    plt.close()
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
    plt.savefig('plots/hists_Rbouy_g_ch.png')
    plt.close()
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
    plt.savefig('plots/hists_Rbouy_r_ch.png')
    plt.close()

    plt.hist(img[:,:,0].ravel(),256,[0,256],color = "blue")
    plt.hist(img[:,:,1].ravel(),256,[0,256],color = "green")
    plt.hist(img[:,:,2].ravel(),256,[0,256],color = "red")
    plt.savefig('plots/hists_Rbouy_total.png')
    plt.close()
    return 0

if __name__ == "__main__":
    yellow_buoy_images = []
    green_buoy_images = []
    red_buoy_images = []
    for file in os.listdir("DataSet/Green"):
        green_buoy_images.append("DataSet/Green/" + file)
    for file in os.listdir("DataSet/Red"):
        red_buoy_images.append("DataSet/Red/" + file)
    for file in os.listdir("DataSet/Yellow"):
        yellow_buoy_images.append("DataSet/Yellow/" + file)

    findHist(yellow_buoy_images, green_buoy_images, red_buoy_images)
    plot3D(yellow_buoy_images, green_buoy_images, red_buoy_images)
