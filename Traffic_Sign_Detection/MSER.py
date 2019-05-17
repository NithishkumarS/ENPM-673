import numpy as np
import cv2



def MSER(inp,new_img):
    cur_image = new_img.copy()
    mser = cv2.MSER_create(_delta = 5, _min_diversity = .7, _max_variation = .2)
    regions, boxes = mser.detectRegions(inp)
    for p in regions:
        xmax, ymax = np.amax(p, axis=0)
        xmin, ymin = np.amin(p, axis=0)
        cv2.rectangle(cur_image, (xmin,ymax), (xmax,ymin), (0, 255, 0), 1)
    return cur_image