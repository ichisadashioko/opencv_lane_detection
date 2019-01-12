import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

frame_idx = 0

def update_frame(cb_value):
    global frame_idx
    frame_idx = int(cv2.getTrackbarPos('frame seek','player'))
    # print(frame_idx)
    # print(cb_value)

def main(argv):
    video_name = ''
    if len(argv) != 2:
        print('Usage: python snap.py <video-name>')
        return
    else:
        video_name = argv[1]

    if not os.path.exists(video_name):
        print('{} not exist'.format(video_name))
        return
    print('''
    Press q or ESC to quit
    Press p to snap the current frame
    ''')

    cap = cv2.VideoCapture(video_name)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cv2.namedWindow('player')
    cv2.createTrackbar('frame seek','player',0,frame_count,update_frame)
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = cap.read()
        
        if ret is False:
            continue

        cv2.imshow('player',frame)

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break
        elif k == ord('p'):
            image_name = '{}-{}.png'.format(video_name,str(frame_idx).zfill(5))
            cv2.imwrite(image_name,frame)


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)