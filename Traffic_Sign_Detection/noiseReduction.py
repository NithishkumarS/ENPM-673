#!/usr/bin/env python3
__author__ = "Nantha Kumar Sunder, Nithish Kumar"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from boundingBox import *
from sklearn.svm import OneClassSVM
import pickle

redSVM = cv2.ml.SVM_load('Models/redSVM.dat')
blueSVM = cv2.ml.SVM_load('Models/blueSVM.dat')
svm = cv2.ml.SVM_load('Models/svm.dat')

def last_4chars(x):
    return(x[-5:])

def loadImages(name):
    imageList = []
    for file in os.listdir(name):
        filename, basename = os.path.splitext(file)
        imageList.append(filename)
    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = name + '/' + str(imageList[i]) + ".jpg"
    return imageList

def resize(img, prop):
    if prop:
        leftTop_x, leftTop_y = int(prop[3]), int(prop[4])
        rightBot_x, rightBot_y = int(prop[5]), int(prop[6])
        img = img[leftTop_y:rightBot_y, leftTop_x:rightBot_x]
    img = cv2.resize(img,(64,64))
    return img

def getHOG():
    signedGradients = True
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 1
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradients)
    return hog

def train_traffic_signs(name):

    hog = getHOG()
    dataset = []
    datalabels = []
    dataCount = 0
    imageCount = 0
    imageList = loadImages(name)
    classId = 1
    while imageCount < len(imageList):
        new_img = cv2.imread(imageList[imageCount])
        new_img = resize(new_img, None)
        des = hog.compute(new_img)
        dataset.append(des)
        datalabels.append(int(1))

        cv2.imshow('frame', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        imageCount = imageCount + 1
        dataCount = dataCount + 1
    cv2.destroyAllWindows()
    svm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    dataset = np.squeeze(np.array(dataset))
    datalabels = np.array(datalabels)
    svm.fit(dataset, datalabels)
    # with open(name, 'wb') as f:
    #     pickle.dump(svm, f)
    # # Set up SVM for OpenCV 3
    # svm = cv2.ml.SVM_create()
    #
    # # Set SVM type
    # svm.setType(cv2.ml.SVM_ONE_CLASS)
    #
    # # Set SVM Kernel to Radial Basis Function (RBF)
    # svm.setKernel(cv2.ml.SVM_POLY)
    #
    # # det degree
    # svm.setDegree(2.5)
    #
    # # Set parameter C
    # svm.setC(3)
    # svm.setP(0)
    # svm.setCoef0(0)
    # svm.setNu(.5)
    #
    # # Set parameter Gamma
    # svm.setGamma(1)
    #
    # # Train SVM on training data
    # dataset = np.squeeze(np.array(dataset))
    # datalabels = np.array(datalabels)
    # svm.train(dataset, cv2.ml.ROW_SAMPLE, datalabels)

    # Save trained model
    # svm.save('Models/'+ name +'.dat')
    print("Training Done")
    return svm

def computeClass(data):
    return svm.predict(data)[1].ravel()
    print('Blur Value',blueSVM.predict(data)[1].ravel())
    print('predict value', svm.predict(data)[1].ravel())
    if redSVM.predict(data)[1].ravel() or blueSVM.predict(data)[1].ravel():
        return svm.predict(data)[1].ravel()
    else:
        return -1

def validateBox(image,corners):
    img = np.copy(image)
    roi = img[corners[1]:corners[3], corners[0]:corners[2]]     # xmin, ymin, xmax, yax
    cv2.rectangle(image, (corners[0], corners[1]), (corners[2], corners[3]), (0,0,255))
    width = 64
    height = 64
    dim = (width, height)
    hog = getHOG()

    resized_img = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)
    des = hog.compute(resized_img)
    dataset = np.squeeze(np.array(des)).reshape((1,-1))
    response = computeClass(dataset)
    print(response)
    if not response == -1:
        text = str(response)
        cv2.putText(image, text, (corners[0], corners[3]+25 ), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (0,0,255))
    return image

def train():
    hog = getHOG()
    dataset = []
    datalabels = []
    folderCount = 0
    folderList = getFolderList("Training")
    dataCount = 0

    while folderCount < len(folderList):
        imageCount = 0
        imageList, prop = loadImages(folderList[folderCount])
        classId = prop[0][-1]
        while imageCount < len(imageList):
            new_img = cv2.imread(folderList[folderCount] + prop[imageCount][0])
            new_img = resize(new_img, None)
            des = hog.compute(new_img)
            dataset.append(des)
            datalabels.append(int(classId))
            # print(int(classId))
#             cv2.imshow('frame', new_img)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
            imageCount = imageCount + 1
            dataCount = dataCount + 1
            # print(imageCount)
        folderCount = folderCount + 1
        cv2.destroyAllWindows()
#     print('dataset', len(dataset))
    # rbf    gamm = 0.9 c =5

    # Set up SVM for OpenCV 3
    svm = cv2.ml.SVM_create()

    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)

    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_RBF)

    # det degree
    # svm.setDegree(2.5)

    # Set parameter C
    svm.setC(12.5)

    # Set parameter Gamma
    svm.setGamma(1.50625)

    # Train SVM on training data
    dataset = np.squeeze(np.array(dataset))
    print(dataset.shape)
    datalabels = np.array(datalabels)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, datalabels)

    # Save trained model
    svm.save("Models/svm.dat")
    print("Training Done")
    return svm


def test(svm, name):
    hog = getHOG()
    dataset = []
    datalabels = []
    dataCount = 0
    imageCount = 0
    imageList = loadImages(name)
    classId = 1
    while imageCount < len(imageList):
        new_img = cv2.imread(imageList[imageCount])
        new_img = resize(new_img, None)
        des = hog.compute(new_img)
        dataset.append(des)
        datalabels.append(int(1))

        cv2.imshow('frame', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        imageCount = imageCount + 1
        dataCount = dataCount + 1
    cv2.destroyAllWindows()

    dataset = np.squeeze(np.array(dataset))
    testResponse = svm.predict(dataset)[1].ravel()
    count = 0
    for i in range(len(testResponse)):
        if (testResponse[i] - datalabels[i]) != 0.0:
            print('Test Value:', testResponse[i])
            print('Actual Value:', datalabels[i])
            count = count + 1

    print(np.array(testResponse))
    print(datalabels)
    percentage = float(len(datalabels)-count)/(len(datalabels))*100
    print('Percentage: ', float(len(datalabels)-count)/(len(datalabels))*100)
    test = np.zeros_like(new_img)
    test = np.array(hog.compute(test)).T
    print(svm.predict(test))

    print('unique responses: ',np.unique(testResponse))
    return percentage

def main():
    """ Main entry point of the app """
    '''
    wins = []
    for i in range(8):
        svm = train_traffic_signs(i)
    #     for i in range(3):
    #         print('current:::::::::::', i)
    #         image = cv2.imread('TSR/Testing/00001/00252_0000'+ str(i)+'.ppm')
    #         validateBox(image, svm)
        res = test(svm,i)
        wins.append(res)
    '''
    '''
#     image = cv2.imread('TSR/Testing/00001/00252_0000'+ str(0)+'.ppm')
    image = cv2.imread('TSR/input/image.032725.jpg')
    corners = [1052,92,1098,158]
    img = validateBox(image, corners)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    '''
    svm = train_traffic_signs('rednoise')
    res = test(svm, 'rednoise')
    svm = train_traffic_signs('bluenoise')
    res = test(svm, 'bluenoise')
    # res = test(svm)

if __name__ == "__main__":
    main()


    '''
    #Visualize model
    x_min, x_max = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    y_min, y_max = datalabels[:, 1].min() - 1, datalabels[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
     np.arange(y_min, y_max, h))

    plt.subplot(1, 1, 1)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(dataset[:, 0], dataset[:, 1], c=datalabels, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.show()
    '''
