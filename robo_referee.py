import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv under python3
import cv2 as cv
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import numpy as np
import argparse
import random as rng
import imutils

class Robo_Referee(object):
    def __init__(self,img_dir,court_w = 540, court_h = 780):
        self.src = cv.imread(img_dir)
        margin = 60
        # h, w = self.src.shape[0], self.src.shape[1]
        # self.src = self.src[:,:int(w/2),:] 
        self.src = self.src[:court_h+margin,:court_w+margin,:]   #crop the image
        self.line_contour = None
        self.ball_loc = None

    def contour_detection(self):
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
            # cv.drawContours(drawing, contours, max_contour_idx, color, 2, cv.LINE_8, hierarchy, 0)
            # # # Show in a window
            # cv.imshow('Contours', drawing)
            return contours[max_contour_idx]

        src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))
        # Create Window
        source_window = 'Source'
        cv.namedWindow(source_window)
        cv.imshow(source_window, self.src)
        max_thresh = 255
        thresh = 74 # initial threshold
        # cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        self.line_contour = thresh_callback(thresh)
        # cv.waitKey()
    
    def ball_detection(self):
        greenLower = (0, 69, 0)
        greenUpper = (79, 255, 255)
        blurred = cv.GaussianBlur(self.src, (11, 11), 0)
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
                cv.circle(self.src, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv.circle(self.src, center, 5, (0, 0, 255), -1)
                self.ball_loc = center
        # cv.imshow("Frame", mask)
        cv.imshow("Frame", self.src)
        cv.waitKey()
        # cv.destroyAllWindows()
    
    def line_judge(self):
        x_array = []
        y_array = []
        for item in self.line_contour:
            x, y = item[0][0],item[0][1]
            x_array.append(x)
            y_array.append(y)
        
        min_idx =0 
        max_idx =0
        min_sum = 1000
        max_sum =0
        for i in range(len(x_array)):
            sum = x_array[i] + y_array[i]
            if sum > max_sum:
                max_sum = sum
                max_idx = i
            if sum < min_sum:
                min_sum = sum
                min_idx = i 


        min_x ,min_y = x_array[min_idx], y_array[min_idx]
        max_x ,max_y = x_array[max_idx], y_array[max_idx]


        ball_x ,ball_y = self.ball_loc[0], self.ball_loc[1]



    #     cv.circle(self.src, (int(min_x), int(min_y)), int(5),
    # (0, 255, 255), 2)
    #     cv.circle(self.src, (int(max_x), int(max_y)), int(5),
    # (0, 255, 255), 2)
    #     cv.imshow("Frame", self.src)
    #     cv.waitKey()

        if ball_x > min_x  and ball_x < max_x and ball_y > min_y and ball_y<max_y:
            return True
        return False




        
img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
robo_referee =Robo_Referee(img_dir)
robo_referee.contour_detection()

robo_referee.ball_detection()
ball_in = robo_referee.line_judge()
print(ball_in)