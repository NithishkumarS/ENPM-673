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
from sklearn.svm import SVC
from skimage.feature import hog
from skimage import data, exposure
import pickle

# redSVM = cv2.ml.SVM_load('Models/redSVM.dat')
# blueSVM = cv2.ml.SVM_load('Models/blueSVM.dat')
# svm = cv2.ml.SVM_load('Models/svm.dat')
with open('Models/mainsvm_data', 'rb') as f:
    svm = pickle.load(f)
# with open('Models/redsvm_data', 'rb') as f:
#     redsvm = pickle.load(f)

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
        print('i:',i)
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

def train_traffic_signs(name):

    hog = getHOG()
    dataset = []
    datalabels = []
    folderCount = 5
    folderList = getFolderList("Training")
    dataCount = 0
    while folderCount < len(folderList):
        imageCount = 0
        imageList, prop = loadImages(folderList[folderCount])
        classId = prop[0][-1]
        while imageCount < len(imageList):
            new_img = cv2.imread(folderList[folderCount] + prop[imageCount][0])
            new_img = resize(new_img, prop[imageCount])
            des = hog.compute(new_img)
            dataset.append(des)
            datalabels.append(int(1))

            cv2.imshow('frame', new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            imageCount = imageCount + 1
            dataCount = dataCount + 1
        folderCount = folderCount + 1
        cv2.destroyAllWindows()

    # Set up SVM for OpenCV 3
    svm = cv2.ml.SVM_create()

    # Set SVM type
    svm.setType(cv2.ml.SVM_ONE_CLASS)

    # Set SVM Kernel to Radial Basis Function (RBF)
    svm.setKernel(cv2.ml.SVM_POLY)

    # det degree
    svm.setDegree(2.5)

    # Set parameter C
    svm.setC(3)
    svm.setP(0)
    svm.setCoef0(0)
    svm.setNu(.5)

    # Set parameter Gamma
    svm.setGamma(1)

    # Train SVM on training data
    dataset = np.squeeze(np.array(dataset))
    datalabels = np.array(datalabels)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, datalabels)

    # Save trained model
    svm.save('Models/'+ name +'.dat')
    print("Training Done")
    return svm

def computeClass(data, mode):
    # return svm.predict(data)[1].ravel()
    # print('rednoise', rednoise.predict(data))
    # print('bluenoise', bluenoise.predict(data))
    feature_arr = ['1','14','17','19','21','35','38','45']
    print('mode', mode)
    print(data.shape)
    val = svm.predict_proba(data)
    maxId = np.amax(val)
    if val[maxId] < 0.5:
        return -1
    else:
        return feature_arr[maxId]
    # print('redsvm', redsvm.predict(data))
    # print('bluesvm', bluesvm.predict(data))
    # # if mode == 2
    # if mode == 2:
    #     if redsvm.predict(data)[0] == '100':
    #         return -1
    #     else:
    #         return redsvm.predict(data)
    # elif mode == 1:
    #     if bluesvm.predict(data)[0] == '100':
    #         return -1
    #     else:
    #         return bluesvm.predict(data)
    # else:
    #     return -1


def validateBox(image,corners, mode):
    img = np.copy(image)
    roi = img[corners[1]:corners[3], corners[0]:corners[2]]     # xmin, ymin, xmax, yax
    width = 64
    height = 64
    dim = (width, height)
    # hog = getHOG()

    resized_img = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)
    # resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    des = hog(resized_img, orientations=32, pixels_per_cell=(4, 4),
            cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L2-Hys')
    print(np.squeeze(des))
    dataset = (np.array(des)).reshape((1,len(des)))
    response = computeClass(dataset, mode)
    print(response)
    if not response == -1:
        text = str(response)
        cv2.rectangle(image, (corners[0], corners[1]), (corners[2], corners[3]), (0,255,0), 2)
        cv2.putText(image, text, (corners[0], corners[3]+25 ), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (0,255,0), 2)
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
    svm.setKernel(cv2.ml.SVM_POLY)

    # det degree
    svm.setDegree(2.5)

    # Set parameter C
    svm.setC(12.5)

    # Set parameter Gamma
    svm.setGamma(0.030625)

    # Train SVM on training data
    dataset = np.squeeze(np.array(dataset))
    print(dataset.shape)
    datalabels = np.array(datalabels)
    svm.train(dataset, cv2.ml.ROW_SAMPLE, datalabels)

    # Save trained model
    svm.save("Models/svm.dat")
    print("Training Done")
    return svm


def test(svm):
    hog = getHOG()
    dataset = []
    datalabels = []
    folderCount = 0
    folderList = getFolderList("Testing")
    dataCount = 0
    imageCount = 0
    '''
    ## Dark image test case --------------------------------------------------------------
    a = np.zeros((64,64,3), dtype = np.uint8)
    b = []
    for i in range(3):

        new_img = resize(a[0], None)
        des = hog.compute(new_img)
        b.append(des)
     '''#----------------------------------------------------------------------------------

    while folderCount < len(folderList):
        print('folder count', folderCount)
        imageCount = 0
        imageList, prop = loadImages(folderList[folderCount])
        if not imageList:
            folderCount = folderCount + 1
            continue
        classId = prop[0][-1]

        while imageCount < len(imageList):
            new_img = cv2.imread(folderList[folderCount] + prop[imageCount][0])
            new_img = resize(new_img, prop[imageCount])
            des = hog.compute(new_img)

            dataset.append(des)
            datalabels.append(int(classId))
            cv2.imshow('frame', new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            imageCount = imageCount + 1
            dataCount = dataCount + 1

        folderCount = folderCount + 1
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
#     svm = train_traffic_signs('blueSVM')
    # svm = train()
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
