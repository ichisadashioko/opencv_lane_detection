import cv2
import numpy as np

cap = cv2.VideoCapture('sample_x4.avi')


def birdview(img, shrink_ratio):
    height, width = img.shape[:2]
    SKYLINE = 85
    src_pts = np.float32([[0, SKYLINE], [width, SKYLINE],
                          [0, height], [width, height]])

    dst_width = 480
    dst_height = 480
    dsize = (dst_width, dst_height)

    x_offset = 30

    dst_pts = np.float32([[0-x_offset, 0], [dst_width+x_offset, 0], [dst_width*shrink_ratio - x_offset, dst_height], [dst_width*(1-shrink_ratio) + x_offset, dst_height]])

    print(dst_pts)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    dst = cv2.warpPerspective(img, M, dsize)

    return dst


while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame.copy()
    height, width = roi.shape[:2]

    cv2.rectangle(roi, (0, 0), (width, 85), 0, -1)

    hls = cv2.cvtColor(roi, cv2.COLOR_RGB2HLS)
    h_channel = hls[:, :, 0]
    cv2.imshow('hue',h_channel)
    l_channel = hls[:, :, 1]
    cv2.imshow('lightness',l_channel)
    s_channel = hls[:, :, 2]
    cv2.imshow('saturation',s_channel)

    # threshold l_channel for white lane lines
    w_ret, w_thresh = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('w_thresh',w_thresh)

    # Sobel both x and y directions
    l_sobel = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
    l_abs_sobel = np.absolute(l_sobel)  # absolute all negative gradient values

    l_sobel_ret, l_sobel_thresh = cv2.threshold(l_abs_sobel, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow('lightness gradient',l_sobel_thresh)

    # Sobel both x and y directions
    s_sobel = cv2.Sobel(s_channel, cv2.CV_64F, 1, 1)
    s_abs_sobel = np.absolute(s_sobel)  # absolute all negative gradient values

    s_sobel_ret, s_sobel_thresh = cv2.threshold(s_abs_sobel, 75, 255, cv2.THRESH_BINARY)
    cv2.imshow('saturation gradient',s_sobel_thresh)

    # mask = cv2.bitwise_and(w_thresh,sobel_thresh)

    for i in range(5):
        # ratio = (i)/10.0
        ratio = (40+i)/100.0
        bird_view = birdview(l_sobel_thresh, ratio)
        cv2.imshow('{}%'.format(int(ratio*100)), bird_view)

    # cv2.imshow('frame',gray)
    if cv2.waitKey(25) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
