import numpy as np
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv under python3
import cv2 as cv
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import numpy as np
import argparse
import random as rng
import imutils
from corner_detection import*
from robo_referee import*
import os



def main():
    # data_path = "/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/HD720_SN14932_16-42-54/"
    front_left_path = "/home/hcl/Documents/ZED/12-2020videos/HD720_SN14932_16-42-54_sync_left/"
    front_right_path = "/home/hcl/Documents/ZED/12-2020videos/HD720_SN14932_16-42-54_sync_right/"
    side_img_path = "/home/hcl/Desktop/GeoVis_Project_Tennis_Tracker/side_img/"

    left_files = os.listdir(front_left_path)
    right_files = os.listdir(front_right_path)
    side_files = os.listdir(side_img_path)
    left_imgs =[]
    right_imgs = []
    side_imgs =[]

    for file in left_files:
        if file.endswith('.png'):
            left_imgs.append(file)
    left_imgs = sorted(left_imgs)

    for file in right_files:
        if file.endswith('.png'):
            right_imgs.append(file)
    right_imgs = sorted(right_imgs)
    
    for file in side_files:
        if file.endswith('.png'):
            side_imgs.append(file)
    side_imgs = sorted(side_imgs)

    frame_num = 1
    robo_referee = Robo_Referee()
    for left_img, right_img, side_img in zip(left_imgs, right_imgs, side_imgs):
        print("frame",frame_num)
        frame_num+=1
        robo_referee.get_image(front_left_path+left_img,side_img_path+side_img)

        #side view
        robo_referee.side_img, robo_referee.side_ball_loc, robo_referee.side_ball_loc_prev, robo_referee.side_velocity = robo_referee.velocity_calc(
            robo_referee.side_img, robo_referee.side_ball_loc, robo_referee.side_ball_loc_prev, robo_referee.greenLower_side, robo_referee.greenUpper_side
            )

        #front view
        robo_referee.apply_homography()
        # robo_referee.get_BEV_transform()
        robo_referee.crop_img()
        robo_referee.contour_detection()
        robo_referee.ball_loc = robo_referee.ball_detection(robo_referee.src,robo_referee.ball_loc,robo_referee.greenLower_front,robo_referee.greenUpper_front)

        if robo_referee.ball_loc != None:
            ball_in = robo_referee.line_judge()
            x , y = robo_referee.ball_loc[0], robo_referee.ball_loc[1]
            if ball_in:
                cv.circle(robo_referee.src, (int(x), int(y)), int(10),
                (0, 255, 0), 2)
                print("Ball is in!!")
            else:
                cv.circle(robo_referee.src, (int(x), int(y)), int(10),
                (0, 0, 255), 2)
                print("Ball is out!")
        else:
            print("No ball is detected")
        
        if cv2.waitKey() == ord('q'):
            break
        # cv.imshow("Orig Frame",robo_referee.orig)
        cv.imshow("Warped Frame", robo_referee.src)
        # cv.imshow("Side Frame",robo_referee.side_img)
        cv.waitKey()
        

main()