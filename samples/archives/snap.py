import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('sample_x4.avi')


def get_mask(img):

    THRESHOLD = 125

    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            delta_0 = (int(img[i][j][0]) - int(img[i][j][1]))**2
            delta_1 = (int(img[i][j][0]) - int(img[i][j][2]))**2
            delta_2 = (int(img[i][j][1]) - int(img[i][j][2]))**2

            delta = delta_0 + delta_1 + delta_2
            if delta < THRESHOLD:
                mask[i][j] = 1
    return mask


def birdview(img, shrink_ratio):
    height, width = img.shape[:2]
    SKYLINE = 85
    src_pts = np.float32([[0, SKYLINE], [width, SKYLINE],
                          [0, height], [width, height]])

    dst_width = 480
    dst_height = 480
    dsize = (dst_width, dst_height)

    x_offset = 30

    dst_pts = np.float32([
        [0-x_offset, 0],
        [dst_width+x_offset, 0],
        [dst_width*shrink_ratio - x_offset, dst_height],
        [dst_width*(1-shrink_ratio) + x_offset, dst_height]])

    # print(dst_pts)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    dst = cv2.warpPerspective(img, M, dsize)

    return dst


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:],axis=0)
    return hist

def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit = np.empty(3)
    right_fit = np.empty(3)
    out_img=np.dstack((img,img,img))*255

    histogram = get_hist(img)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)


while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 127, 255)
    cv2.imshow('edges', edges)

    roi = frame.copy()
    height, width = roi.shape[:2]

    cv2.rectangle(roi, (0, 0), (width, 90), 0, -1)  # remove the upper region

    hls = cv2.cvtColor(roi, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    cv2.imshow('hue', h_channel)
    l_channel = hls[:, :, 1]
    cv2.imshow('lightness', l_channel)
    s_channel = hls[:, :, 2]
    cv2.imshow('saturation', s_channel)

    # threshold l_channel for white lane lines
    w_ret, w_thresh = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('w_thresh', w_thresh)

    # Sobel both x and y directions
    l_sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    l_abs_sobel = np.absolute(l_sobel)  # absolute all negative gradient values

    l_sobel_ret, l_sobel_thresh = cv2.threshold(
        l_abs_sobel, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow('lightness gradient', l_sobel_thresh)

    # Sobel both x and y directions
    s_sobel = cv2.Sobel(s_channel, cv2.CV_64F, 1, 1)
    s_abs_sobel = np.absolute(s_sobel)  # absolute all negative gradient values

    s_sobel_ret, s_sobel_thresh = cv2.threshold(
        s_abs_sobel, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow('saturation gradient', s_sobel_thresh)

    # mask = cv2.bitwise_and(w_thresh,sobel_thresh)

    ratio = (40+4)/100.0
    bird_view = birdview(l_sobel_thresh, ratio)
    cv2.imshow('lightness gradient {}% sky view'.format(
        int(ratio*100)), bird_view)

    cv2.imshow('sky view', birdview(frame, ratio))

    # enhanced_mask = cv2.bitwise_and(l_sobel_thresh,get_mask(frame))
    # enhanced_mask = get_mask(frame)*255
    # cv2.imshow('enchanced mask',enhanced_mask)

    k = cv2.waitKey(25)
    # print(k)
    if k & 0xff == ord('p'):
        cv2.imwrite('sample.png',frame)
        plt.imshow(img)
        plt.show()
    if k & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
