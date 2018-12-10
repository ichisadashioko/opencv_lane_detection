#!/usr/bin/env python
# shebang (#!)
import sys,time
from scipy.ndimage import filters # scipy is with with np and cv to handle conversions

import cv2
import numpy as np

import rospy
import roslib

from sensor_msgs.msg import CompressedImage


VERBOSE=True # comment for tinkering

class image_feature:
    def __init__(self):
        self.image_pub = rospy.Publisher('ouput/image_raw/compressed',CompressedImage,queue_size=1)

        self.subscriber = rospy.Subscriber('/Team1_image/compressed',CompressedImage,self.callback,queue_size=1)

        if VERBOSE:
            print("subscribed to /Team1_image/compressed")

    def callback(self,ros_data):
        # if VERBOSE:
        #     print('received image of type: "%s"' % ros_data.format)

        # decode image
        np_arr = np.fromstring(ros_data.data,np.uint8)
        # img = cv2.imdecode(np_arr,cv2.CV_LOAD_IMAGE_COLOR)
        img = cv2.imdecode(np_arr,cv2.IMREAD_COLOR) # OpenCV >= 3.0

        # Feature detectors using CV2
        # "","Grid","Pyramid" +
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        method = "GridFAST"
        # feat_det = cv2.FeatureDetector_create(method)
        feat_det = cv2.FastFeatureDetector_create()
        time1 = time.time()

        # convert np image to grayscale
        # featPoints = feat_det.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        featPoints = feat_det.detect(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        time2 = time.time()

        if VERBOSE:
            print('%s detector found: %s points in %s sec.'%(method,len(featPoints),time2-time1))

        for featpoint in featPoints:
            x,y = featpoint.pt
            cv2.circle(img,(int(x),int(y)),1,(0,0,255),-1)

        cv2.imshow('cv_img',img)
        cv2.waitKey(2)

        # create CompressedImage
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = 'jpeg'
        msg.data = np.array(cv2.imencode('.jpg',img)[1]).tostring()

        # publish new image
        self.image_pub.publish(msg)

def myhook():
    print "shutdown time!"

def main(args):
    ic = image_feature()
    rospy.init_node('image_feature',anonymous=True)
    rospy.on_shutdown(myhook)
    try:
        # while not rospy.is_shutdown():
        rospy.spin()
    except KeyboardInterrupt:
        print 'Shutting down ROS Image feature detector module'
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)