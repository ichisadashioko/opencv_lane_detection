import cv2
import numpy as np

cap = cv2.VideoCapture('/home/shioko/Videos/20181211_110224.avi')

def birdview(img,shrink_ratio):
    height,width = img.shape[:2]
    SKYLINE = 85
    src_pts = np.float32([[0,SKYLINE],[width,SKYLINE],[0,height],[width,height]])

    dst_width = width
    dst_height = height - SKYLINE + 120
    dsize = (dst_width,dst_height)

    dst_pts = np.float32([[0,0],[dst_width,0],[dst_width*shrink_ratio,dst_height],[dst_width*(1-shrink_ratio),dst_height]])

    M = cv2.getPerspectiveTransform(src_pts,dst_pts)

    dst = cv2.warpPerspective(img,M,dsize)

    return dst

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    for i in range(5):
        ratio = (i)/10.0
        bird_view = birdview(frame,ratio)
        cv2.imshow('{}%'.format(int(ratio*100)),bird_view)

    cv2.imshow('frame',gray)
    if cv2.waitKey(25) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()