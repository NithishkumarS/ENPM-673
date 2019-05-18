import numpy as np
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from boundingBox import *

def getInp(new_img):
    mask_blue, mask_red = colorSegmentation(new_img)
    # kernel = np.ones((3,3),np.uint8)
    # erosion = cv2.erode(mask_blue,kernel,iterations = 1)
    # erosion = cv2.Canny(erosion,100,200)
    # blackhat = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)
    imb = cv2.cvtColor(mask_blue, cv2.COLOR_BGR2GRAY)

    imr = cv2.cvtColor(mask_red, cv2.COLOR_BGR2GRAY)
    im = imb | imr
    return imb, imr

# def hisEqulColor(img):
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
#     channels = cv2.split(ycrcb)
#     cv2.equalizeHist(channels[0], channels[0])
#     cv2.merge(channels, ycrcb)
#     cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
#     return img

def masker(img):
    img[int(img.shape[0]/2):,:] = 0
    return img

def contrastNormalize(new_img):
    r_channel = new_img[:,:,2]
    g_channel = new_img[:,:,1]
    b_channel = new_img[:,:,0]

    arr = np.asarray(r_channel)
    r_channel = imadjust(arr,arr.min(),arr.max(),0,1).astype(np.float32)
    arr = np.asarray(g_channel)
    g_channel = imadjust(arr,arr.min(),arr.max(),0,1).astype(np.float32)
    arr = np.asarray(b_channel)
    b_channel = imadjust(arr,arr.min(),arr.max(),0,1).astype(np.float32)

    sum = r_channel + g_channel + b_channel
    sum[sum==0] = 1

    r_channel = np.divide(r_channel, sum)
    g_channel = np.divide(g_channel, sum)
    b_channel = np.divide(b_channel, sum)

    imr = np.maximum(0,np.minimum((r_channel-b_channel),(r_channel-g_channel)))
    imb = np.maximum(0,b_channel-r_channel)#,(b_channel-g_channel)))

    return imr, imb

def imadjust(x,a,b,c,d,gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def boundingBox_mser(new_img):
    imr, imb = contrastNormalize(new_img)
    # new_img = MSER(imr, new_img)
    new_img = MSER(imb, new_img)
    return new_img

def MSER(img, new_img):
    img = masker(img)
    img = np.uint8(img)
    new_img = new_img.copy()
    mser = cv2.MSER_create(_delta = 2, _min_diversity = 0.8, _max_variation = .2)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(img)
    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        # w = xmax - xmin
        # h = ymax - ymin
        # if h >= 0.9*w and w*h > 100 and (h < 2.5*w) and w*h < 30000:
        #     cv2.rectangle(new_img, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
        cv2.rectangle(new_img, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    return new_img
