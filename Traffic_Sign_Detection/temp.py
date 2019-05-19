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

redSVM = cv2.ml.SVM_load('Models/redSVM.dat')
blueSVM = cv2.ml.SVM_load('Models/blueSVM.dat')
svm = cv2.ml.SVM_load('Models/svm.dat')
with open('Models/rednoise_data', 'rb') as f:
    rednoise = pickle.load(f)
with open('Models/bluenoise_data', 'rb') as f:
    bluenoise = pickle.load(f)

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
    basename_final = []
    for file in os.listdir(str):
        filename, basename = os.path.splitext(file)
        imageList.append(filename)
        basename_final = basename
    imageList = sorted(imageList, key = last_4chars)
    for i in range(len(imageList)):
        imageList[i] = str + '/' + imageList[i] + basename_final
    return imageList

def getImages(add):
    prop = []
    property = []
    for file in os.listdir(add):
        filename, basename = os.path.splitext(file)
        if basename == '.csv':
            prop.append(file)
            break
    with open(add + prop[0],'r') as f:
        property = list(csv.reader(f, delimiter=";"))
    property = property[1:]
    return property

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

def traffic_signs(name, mode, dataset, datalabels):

    # hog = getHOG()
    # dataset = []
    # datalabels = []
    total_len = 0
    folderList = getFolderList(name)
    if mode == 1:
        folderCount = 5
        total_len = len(folderList)
    elif mode == 2:
        folderCount = 0
        total_len = len(folderList) - 3
    else:
        folderCount = 0
        total_len = len(folderList)

    dataCount = 0
    while folderCount < total_len:
        imageCount = 0
        prop = getImages(folderList[folderCount])
        classId = prop[0][-1]
        while imageCount < len(prop):
            new_img = cv2.imread(folderList[folderCount] + prop[imageCount][0])
            new_img = resize(new_img, prop[imageCount])
            # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            fd, hog_image = hog(new_img, orientations=32, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L2-Hys')
            # des = hog.compute(new_img)
            # hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            # cv2.imshow('hog_image_rescaled', hog_image_rescaled)
            # cv2.waitKey(0)
            # print(len(fd))
            dataset.append(fd)
            datalabels.append(classId)
            # cv2.imshow('frame', new_img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            imageCount = imageCount + 1
            dataCount = dataCount + 1
        folderCount = folderCount + 1
        # cv2.destroyAllWindows()
    print("Dataset Taken")
    return dataset, datalabels

def noisedata(dataset, datalabels, name):
    hog = getHOG()
    imageCount = 0
    imageList = loadImages(name)
    classId = 100
    while imageCount < len(imageList):
        new_img = cv2.imread(imageList[imageCount])
        new_img = resize(new_img, None)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
        des = hog(new_img, orientations=32, pixels_per_cell=(4, 4),
                cells_per_block=(1, 1), visualize=True, multichannel=True, block_norm='L2-Hys')
        dataset.append(des)
        datalabels.append(classId)

        cv2.imshow('frame', new_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        imageCount = imageCount + 1
    cv2.destroyAllWindows()

    print("Noise Dataset Taken")
    return dataset, datalabels

def computeClass(data, mode):
    # return svm.predict(data)[1].ravel()
    print('rednoise', rednoise.predict(data))
    print('bluenoise', bluenoise.predict(data))
    print('mode', mode)
    print('Predict', svm.predict(data)[1].ravel())
    # if mode == 2
    if (rednoise.predict(data) == -1 and mode == 2):
        return svm.predict(data)[1].ravel()
    elif (bluenoise.predict(data)==-1 and mode == 1):
        return svm.predict(data)[1].ravel()
    else:
        return -1


def validateBox(image,corners, mode):
    img = np.copy(image)
    roi = img[corners[1]:corners[3], corners[0]:corners[2]]     # xmin, ymin, xmax, yax
    width = 64
    height = 64
    dim = (width, height)
    hog = getHOG()
    resized_img = cv2.resize(roi, dim, interpolation = cv2.INTER_AREA)
    des = hog.compute(resized_img)
    dataset = np.squeeze(np.array(des)).reshape((1,-1))
    response = computeClass(dataset, mode)
    print(response)
    if not response == -1:
        text = str(response)
        cv2.rectangle(image, (corners[0], corners[1]), (corners[2], corners[3]), (0,255,0), 2)
        cv2.putText(image, text, (corners[0], corners[3]+25 ), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (0,255,0), 2)
    return image

def train(dataset, datalabels, name, class_weight):
    dataset = np.squeeze(np.array(dataset))
    datalabels = np.array(datalabels).T
    weights = datalabels.copy()

    # weights[weights != '100'] = 1
    # weights[weights == '100'] = -100
    svm = SVC(C=1,gamma=0.001, class_weight=class_weight, probability = True)
    svm.fit(dataset, datalabels)
    with open('Models/' + name + 'svm_data', 'wb') as f:
        pickle.dump(svm, f)
    print("Training Done")
    return svm

def test(svm, dataset, datalabels):
    dataset = np.squeeze(np.array(dataset))
    datalabels = np.array(datalabels).T
    testResponse = svm.predict(dataset)
    count = 0
    for i in range(len(testResponse)):
        if (testResponse[i] != datalabels[i]):
            count = count + 1

    # print(np.array(testResponse))
    # print(datalabels)
    percentage = float(len(datalabels)-count)/(len(datalabels))*100
    print('Percentage: ', float(len(datalabels)-count)/(len(datalabels))*100)

    print('unique responses: ',np.unique(testResponse))
    return percentage

def main():
    """ Main entry point of the app """
    class_weight_blue = {'45': 100.,
                    '38': 1.,
                    '35': 1.,
                    '100': 1
                    }
    class_weight_red = {'21': 1.,
                    '19': 1.,
                    '17': 1.,
                    '14': 1.,
                    '1': 1.,
                    '100': 1
                    }
    class_weight_red = 'balanced'
    class_weight_blue = 'balanced'
    dataset = []
    datalabels = []
    dataset, datalabels = traffic_signs('Training', 1, dataset, datalabels)
    # print(len(dataset))
    # dataset, datalabels = noisedata(dataset, datalabels , 'bluenoise')
    # svm = train(dataset, datalabels, 'blue', class_weight_blue)
    # res = test(svm, dataset, datalabels)
    # dataset, datalabels = traffic_signs('Testing', 1)
    # res = test(svm, dataset, datalabels)

    dataset, datalabels = traffic_signs('Training', 2, dataset, datalabels)
    # print(len(dataset))
    # dataset, datalabels = noisedata(dataset, datalabels , 'rednoise')
    svm = train(dataset, datalabels, 'main', class_weight_red)
    res = test(svm, dataset, datalabels)
    dataset = []
    datalabels = []
    dataset, datalabels = traffic_signs('Testing', 3, dataset, datalabels)
    res = test(svm, dataset, datalabels)


if __name__ == "__main__":
    main()
