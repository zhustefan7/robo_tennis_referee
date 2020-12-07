import torch
import argparse
import os
import numpy as np
import imutils


import cv2 as cv
import time


def ball_detection(src):
    greenLower = (0, 69, 0)
    greenUpper = (79, 255, 255)
    blurred = cv.GaussianBlur(src, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    print(hsv.shape)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)
    # cv.imshow("Frame", output)

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv.contourArea)
        # cv.drawContours(self.src,c, -1, (0,255,0), 3)


        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv.circle(src, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv.circle(src, center, 5, (0, 0, 255), -1)
            ball_loc = center
    # cv.imshow("Frame", mask)
    cv.imshow("Frame", src)
    cv.waitKey(1)
    

video_name = "20201206_164251"
video_path = "/home/hcl/Desktop/GeoVis_Project_Tennis_Tracker/20201206_164251.mp4"
# video_path = "side.mp4"
cap = cv.VideoCapture(video_path)
frame_num = 0
if (cap.isOpened()== False):
  print("Error opening video stream or file")
 
ret = True
while ret == True:
    ret, frame = cap.read()    
    frame_num += 1
    print(frame_num)
    # cv2.imshow('Frame', frame)
    
    
    # ball_detection(frame)

    # save frames
    if False:
        cv.imwrite(video_name+"_"+"{:03d}".format(frame_num)+".png",frame)
    
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

 
cap.release()
cv.destroyAllWindows()