import numpy as np
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
from boundingBox import *
from trafficSignClassification import *

def blueSeg(image):
    '''
        function to color segment for blue color using hsv
        input: mask image
        output: mask image
    '''

    hsv = cv2.cvtColor(image,  cv2.COLOR_BGR2HSV).astype(np.float)

    #--------------------- Blue mask ---------------------------
    lower_blue = np.array([100,100,100])
    upper_blue = np.array([150,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((8,8),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # flood fill background to find inner holes
    holes = mask.copy()
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with img fill in holes
    holes = cv2.bitwise_not(holes)
    mask = cv2.bitwise_or(mask, holes)

    blue_mask = cv2.bitwise_and(image,image, mask= mask)
    #-----------------------------------------------------------

    return blue_mask

def redSeg(image):
    '''
        function to color segment for red color using hsv
        input: mask image
        output: mask image
    '''

    hsv = cv2.cvtColor(image,  cv2.COLOR_BGR2HSV).astype(np.float)

    # -------------------- Red mask ----------------------------
    # lower mask
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask_light_red = cv2.inRange(hsv, lower_red, upper_red)

    # upper mask
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask_dark_red = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask_light_red + mask_dark_red

    holes = mask.copy()
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with img fill in holes
    holes = cv2.bitwise_not(holes)
    mask = cv2.bitwise_or(mask, holes)

    # combining mask
    red_mask = cv2.bitwise_and(image, image, mask = mask)

    return red_mask

def masker(img):
    img[int(img.shape[0]/2):,:] = 0
    return img

def contrastNormalize(new_img):
    '''
        function to find the normalised intensity for the red and blue
        input: Denoised image
        output: Normalised Red Image and Normalised Blue Image
    '''

    # Getting the channels of R, B and G
    r_channel = new_img[:,:,2]
    g_channel = new_img[:,:,1]
    b_channel = new_img[:,:,0]

    # Stretching it based on lower and upper limit for all the channels
    arr = np.asarray(r_channel)
    r_channel = contrastStretch(arr,arr.min(),arr.max(),0,1).astype(np.float32)
    arr = np.asarray(g_channel)
    g_channel = contrastStretch(arr,arr.min(),arr.max(),0,1).astype(np.float32)
    arr = np.asarray(b_channel)
    b_channel = contrastStretch(arr,arr.min(),arr.max(),0,1).astype(np.float32)

    # fidn the sum of the channels
    sum = r_channel + g_channel + b_channel
    sum[sum==0] = 1

    # normalizing tht image
    r_channel = np.divide(r_channel, sum)
    g_channel = np.divide(g_channel, sum)
    b_channel = np.divide(b_channel, sum)

    # Contrast Normailzing the red Channel
    imr = np.maximum(0,np.minimum((r_channel-b_channel),(r_channel-g_channel)))

    # Contrast Normalizing the blue channel
    imb = np.maximum(0,b_channel-r_channel)#,(b_channel-g_channel)))

    return imr, imb

def contrastStretch(x,a,b,c,d):
    y = (((x - a) / (b - a))) * (d - c) + c
    return y

def boundingBox_mser(new_img):
    '''
        function to Find the bounding box
        input: Denoise Image
        output: image with traffic sign information
    '''

    # Contrast Normalizing the image
    imr, imb = contrastNormalize(new_img)
    # cv2.imshow('imr', imb)
    # cv2.waitKey(0)

    # finding the red traffic signs
    bounded_img, corners = MSER(imr, new_img, 2)

    for corner in corners:
        new_img = validateBox(new_img, corner, 2)

    # Finding the blue traffic signs
    bounded_img, corners = MSER(imb, new_img, 1)
    # cv2.imwrite('red_bounded_img.png',bounded_img)
    # print(corners)
    for corner in corners:
        new_img = validateBox(new_img, corner, 1)
    # cv2.imwrite('Output.png',new_img)

    return new_img

def MSER(img, new_img, mode):
    '''
        function to find MSER
        Input: Constrast Image, Original Image, Mode (1 or 2)
        Output: Original Image with bounding box, corners points of potential traffic sign
    '''
    # Pre process
    img = masker(img)
    img = img*255
    img = img.astype(np.uint8)
    # cv2.imwrite('red_contrast.png',img)
    new_img = new_img.copy()

    # Find MSER
    mser = cv2.MSER_create(_delta = 4, _min_diversity = 0.8, _max_variation = .2)
    regions, boxes = mser.detectRegions(img)
    corners = []
    # Adding a mask for all found points
    mask = np.zeros_like(img)
    for points in regions:
        for point in points:
            mask[point[1],point[0]] = 255

    # Processing the mask usign the hsv image threshold
    masked_image = cv2.bitwise_and(new_img, new_img, mask = mask)
    # cv2.imwrite('red_masked_image_MSER.png',masked_image)
    if mode==1:
        masked_image=blueSeg(masked_image)
    elif mode==2:
        masked_image=redSeg(masked_image)

    # cv2.imwrite('red_masked_image_segmented.png',masked_image)
    # Finding the corners points of the potential bounded box
    im = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(im, 5, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if h >= 0.9*w and w*h > 1000 and (h < 2.0*w) and w*h < 30000:
            corners.append([x,y,x+w,y+h])
            cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),2)

    return new_img, corners
