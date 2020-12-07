import torch
import argparse
import os
import numpy as np

import cv2
import time




video_path = "/home/hcl/Desktop/GeoVis_Project_Tennis_Tracker/20201206_164251.mp4"
cap = cv2.VideoCapture(video_path)
frame_num = 0
if (cap.isOpened()== False):
  print("Error opening video stream or file")
 
while (cap.isOpened()):
    ret, frame = cap.read()
    frame_num += 1
    print(frame_num)
    if ret == True:
        cv2.imshow('Frame', frame)
 
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
 
    else:
        break
 
cap.release()
cv2.destroyAllWindows()