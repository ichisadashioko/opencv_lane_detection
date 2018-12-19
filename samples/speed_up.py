import cv2
import os
import numpy


cap = cv2.VideoCapture('sample.avi')

file_name = os.path.join(os.getcwd(),'sample_x4.avi')

fps = 40
frame_size = (320,240)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

video_writer = cv2.VideoWriter(file_name,fourcc,fps,frame_size)

idx = 0

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    idx += 1
    if idx % 4 == 0:
        video_writer.write(frame)

cap.release()
video_writer.release()