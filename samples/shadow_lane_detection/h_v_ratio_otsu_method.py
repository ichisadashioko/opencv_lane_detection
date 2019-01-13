import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import time

import matplotlib.pyplot as plt

def get_shadow_region(img):
    ''' Find the shadow region of the image
    Parameter
    ---------
    img : numpy.ndarray
        The BGR image
    '''
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    h_channel = hsv[:,:,0].astype(np.float64)
    s_channel = hsv[:,:,1].astype(np.float64)
    v_channel = hsv[:,:,2].astype(np.float64)

    h_v_ratio = ((h_channel+1) / (v_channel+1)).astype(np.uint8)
    ret, shadow = cv2.threshold(h_v_ratio,0,255,cv2.THRESH_OTSU)
    return shadow

def main():
    for i in range(4):
        img = cv2.imread('shadow_0{}.png'.format(i+1))
        
        shadow = get_shadow_region(img)

        color_img = np.zeros(img.shape, np.uint8)
        color_img[:,:] = (75,75,0)
        color_mask = cv2.bitwise_and(color_img,color_img,mask=shadow)
        masked_img = cv2.addWeighted(img,1,color_mask,1,0)
        cv2.imshow('color_mask',masked_img)

        cv2.imshow('img',img)
        cv2.imshow('thresh',shadow)
        cv2.waitKey()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()