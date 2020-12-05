import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import cv2 as cv
import numpy as np


import argparse
import random as rng
rng.seed(12345)
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
    max_contour_len = 0
    max_contour_idx = 0

    for i in range(len(contours)):
        if (len(contours[i])) > max_contour_len:
            max_contour_len = len(contours[i])
            max_contour_idx = i
        # print(len(contours[i]))
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv.drawContours(drawing, contours, max_contour_idx, color, 2, cv.LINE_8, hierarchy, 0)
    # Show in a window
    print(contours[max_contour_idx])
    cv.imshow('Contours', drawing)
# Load source image
img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
src = cv.imread(img_dir)



if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()