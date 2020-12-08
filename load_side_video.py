import argparse
import os
import numpy as np
import imutils


import cv2 as cv
import time

def ball_detection(src):
    #for serving    
    # greenLower = (0, 65, 154)
    # greenUpper = (63, 255, 255)
    
    #for side view
    greenLower = (0, 64, 142)
    greenUpper = (95, 255, 255)
    blurred = cv.GaussianBlur(src, (11, 11), 0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
    # print(hsv.shape)
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
    # cv.imshow("Frame", src)
    # cv.waitKey(1)
    return center,src
    

use_vid = True

if use_vid:
    # video_path = "/home/hcl/Documents/ZED/11-2020_videos/HD1080_SN14932_16-37-43.avi"
    video_path = "/home/hcl/Documents/ZED/11-2020_videos/HD1080_SN14932_16-31-32.avi"
    cap = cv.VideoCapture(video_path)
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

#detect ball with images
side_img_path = "/home/hcl/Desktop/GeoVis_Project_Tennis_Tracker/side_img/"
files = os.listdir(side_img_path)
imgs =[]
labels =[]

for file in files:
    if file.endswith('.png'):
        imgs.append(file)
imgs = sorted(imgs)

fps = 60
scale = 10
ball_loc_prev = None
ball_loc = None
contact_loc = None
i = 0



ret = True
while ret == True:
    if use_vid:
        print("frame",i)
        ret, side_img = cap.read()
        i += 1    
    
    else:
        # for img in imgs:
        side_img = cv.imread(side_img_path+imgs[i])
        i+=1
            
    side_img = cv.rotate(side_img, cv.ROTATE_90_CLOCKWISE)
    side_img = cv.resize(side_img,(int(side_img.shape[1]/2),int(side_img.shape[0]/2)))
    ball_loc,side_img = ball_detection(side_img)
    # print("ball loc",ball_loc)
    # print("prev ball loc", ball_loc_prev)
    
    print(ball_loc)
    #calculate velocity vector and plot
    if ball_loc != None and ball_loc_prev != None:
        velocity_vec = tuple(map(lambda i, j: (i - j), ball_loc, ball_loc_prev))
        # print(velocity_vec)
        print("velocity:",np.linalg.norm(np.array(velocity_vec)/1/60))
        side_img = cv.arrowedLine(
            side_img,
            pt1=ball_loc,
            pt2=tuple(map(lambda i, j: i + j*scale, ball_loc, velocity_vec)),
            color=(255,0,0), thickness=3) 

        # frame when ball touches ground
        if velocity_vec[0] == 0:
            velocity_slope = np.inf
        else:
            velocity_slope = velocity_vec[1]/velocity_vec[0]
        # print(velocity_slope)
        if contact_loc == None and velocity_slope >= 0:
            contact_loc = ball_loc_prev
        if contact_loc != None:
            cv.circle(side_img, contact_loc, 10, (0, 0, 255),-1)
        # print(contact_loc)

    ball_loc_prev = ball_loc

    cv.imshow("Frame", side_img)
    cv.waitKey()

    if cv.waitKey(25) & 0xFF == ord('q'):
        break





#load video, save frames
if False:
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
        
        # save frames
        if True:
            cv.imwrite(video_name+"_"+"{:03d}".format(frame_num)+".png",frame)
        
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    
    cap.release()
    cv.destroyAllWindows()