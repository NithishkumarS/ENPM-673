#!/usr/bin/env python3
__author__ = "Nantha Kumar Sunder, Nithish Kumar"
__version__ = "0.1.0"
__license__ = "MIT"

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import csv
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from boundingBox import *

def last_4chars(x):
    return(x[-5:])

def getFolderList(fol_str):
    folderList = []
    for file in os.listdir("TSR/" + fol_str + "/"):
        filename, basename = os.path.splitext(file)
        if not basename:
            folderList.append(filename)
    folderList = sorted(folderList, key = last_4chars)
    for i in range(len(folderList)):
        folderList[i] ="TSR/" + fol_str + "/" + folderList[i] + "/"

    return folderList

def loadImages(str):
    imageList = []
    prop = []
    for file in os.listdir(str):
        filename, basename = os.path.splitext(file)
        if basename == '.ppm':
            imageList.append(filename)
        if basename == '.csv':
            prop.append(file)
    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = str + imageList[i] + ".ppm"
    with open(str + prop[0],'r') as f:
        property = list(csv.reader(f, delimiter=";"))
    property = property[1:]
    return imageList, property

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
            cv2.imshow('frame', new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            imageCount = imageCount + 1
            dataCount = dataCount + 1
            # print(imageCount)
        folderCount = folderCount + 1
        cv2.destroyAllWindows()

    # Set up SVM for OpenCV 3
    svm = cv2.ml.SVM_create()

    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)

    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_POLY)

    # det degree
    svm.setDegree(2.5)

    # Set parameter C
    svm.setC(2.5)

    # Set parameter Gamma
    svm.setGamma(0.030625)

    # Train SVM on training data
    dataset = np.squeeze(np.array(dataset))
    print(dataset.shape)
    datalabels = np.array(datalabels)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, datalabels)

    # Save trained model
    svm.save("svm_model.yml")

    # cross validation

    print("Training Done")

    return svm

def validateBox(image, svm):
    img = np.copy(image)
    width = 64
    height = 64
    dim = (width, height)
    hog = getHOG()
    
    resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
    des = hog.compute(resized_img)
    dataset = np.squeeze(np.array(des))
    testResponse = svm.predict(dataset)[1].ravel()
    print('respose', testResponse)

#     return points

def test(svm):

    hog = getHOG()
    dataset = []
    datalabels = []
    folderCount = 0
    folderList = getFolderList("Testing")
    dataCount = 0
    while folderCount < len(folderList):
        imageCount = 0
        imageList, prop = loadImages(folderList[folderCount])
        if not imageList:
            folderCount = folderCount + 1
            continue
        # print(prop)
        print(folderList[folderCount])
        classId = prop[0][-1]

        while imageCount < len(imageList):
            new_img = cv2.imread(folderList[folderCount] + prop[imageCount][0])
            new_img = resize(new_img, None)
            des = hog.compute(new_img)
            dataset.append(des)
            datalabels.append(int(classId))
            cv2.imshow('frame', new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            imageCount = imageCount + 1
            dataCount = dataCount + 1
            # print(imageCount)
        folderCount = folderCount + 1
        cv2.destroyAllWindows()
    dataset = np.squeeze(np.array(dataset))
    testResponse = svm.predict(dataset)[1].ravel()
    count = 0
    for i in range(len(testResponse)):
        if (testResponse[i] - datalabels[i]) != 0.0:
            print('Test Value:', testResponse[i])
            print('Actual Value:', datalabels[i])
            print(i)
            count = count + 1


    print(np.array(testResponse))
    print(datalabels)
    print('Percentage: ', float(len(datalabels)-count)/(len(datalabels))*100)
    print(np.unique(testResponse))
    return 0

def main():
    """ Main entry point of the app """
    svm = train()
    image = cv2.imread('TSR/Testing/00001/00252_00000.ppm')
    validateBox(image, svm)
#     test(svm)
    # test()

if __name__ == "__main__":
    main()
