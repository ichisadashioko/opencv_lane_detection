#!/usr/bin/env python
import sys,time
import os
from datetime import datetime

import numpy as np
import cv2

import rospy
import roslib

from sensor_msgs.msg import CompressedImage

usr = os.path.expanduser('~') # type str
path = os.path.join(usr,'Videos')

now = datetime.now()
timestamp = '{}{}{}_{}{}{}'.format(str(now.year).zfill(4),str(now.month).zfill(2),str(now.day).zfill(2),str(now.hour).zfill(2),str(now.minute).zfill(2),str(now.second).zfill(2))

file_name = os.path.join(path,timestamp + '.avi')

fps = 40
frame_size = (320,240)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_writer = cv2.VideoWriter(file_name,fourcc,fps,frame_size)

class ImageSub:
    def __init__(self):

        self.subscriber = rospy.Subscriber('/Team1_image/compressed',CompressedImage,self.callback,queue_size=1)

    def callback(self,ros_data):
        global video_writer

        np_arr = np.fromstring(ros_data.data,np.uint8)
        img = cv2.imdecode(np_arr,cv2.IMREAD_COLOR) # OpenCV >= 3.0
        # img = cv2.imdecode(np_arr,cv2.CV_LOAD_IMAGE_COLOR)
        # if is_record:
        #     img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        video_writer.write(img)
        cv2.imshow('img',img)
        cv2.waitKey(2)


def shutdown_hook():
    video_writer.release()
    cv2.destroyAllWindows()
    print('ros node is shutting down')

def main(args):
    node = ImageSub()
    rospy.init_node('video_recorder',anonymous=True)
    rospy.on_shutdown(shutdown_hook)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)