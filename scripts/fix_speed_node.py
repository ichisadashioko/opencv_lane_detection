#!/usr/bin/env python
from __future__ import print_function
import sys
import time

import cv2
import numpy as np

import rospy
import roslib

from sensor_msgs.msg import CompressedImage
import std_msgs
from std_msgs.msg import Float32

def pub_speed():
    pub = rospy.Publisher('Team1_speed', Float32,queue_size=1)
    rospy.init_node('pub_speed',anonymous=True)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        pub.publish(50)
        rate.sleep()

if __name__ == '__main__':
    try:
        pub_speed()
    except rospy.ROSInterruptException:
        pass