import numpy as np
import os, sys
# This try-catch is a workaround for Python3 when used with ROS; it is not needed for most platforms
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
colors = ['Red', 'Yellow', 'Green']
for j in range(0,3):
    xtrain = np.array([0,0,0])
    for file in os.listdir('DataSet/' + colors[j] + '/'):
        basename, extension = os.path.splitext(file)
        img = cv2.imread('DataSet/'+ colors[j] + '/'+basename+'.jpg')
        for i in range(img.shape[0]):
            xtrain= np.vstack((xtrain, img[i,:]))
    xtrain = np.delete(xtrain, 0, 0)
    print(j)
    print('xtrain_' + colors[j])
    np.save('xtrain_' + colors[j], xtrain)
