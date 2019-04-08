import numpy as np
import os, sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
colors = ['Red', 'Yellow', 'Green']
for j in range(0,3):
    channel = np.array([0,0,0])
    for file in os.listdir('DataSet/' + colors[j] +'/'):
        basename, extension = os.path.splitext(file)
        img = cv2.imread('DataSet/'+ colors[j] + '/'+basename+'.jpg')
        for i in range(img.shape[0]):
            channel= np.vstack((channel, img[i,:]))
    channel = np.delete(channel, 0, 0)
    print(j)
    print('channel_' + colors[j])
    np.save('Input/channel_' + colors[j], channel)
