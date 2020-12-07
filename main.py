import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv under python3
import cv2 as cv
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import numpy as np
import argparse
import random as rng
import imutils
from corner_detection import*
from robo_referee import*
import os



def main():
    data_path = "/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/HD720_SN14932_16-42-54/"
    files = os.listdir(data_path)
    imgs =[]
    labels =[]

    for file in files:
        if file.endswith('.png'):
            imgs.append(file)
    imgs = sorted(imgs)

    for file in files:
        if file.endswith('.txt'):
            labels.append(file)
    imgs = sorted(imgs)
    labels = sorted(labels)



    # Need to Change this number for different classes
    class_label =0
    frame_num = 1

    print(imgs)
    robo_referee =Robo_Referee()
    for img in imgs:
        robo_referee.get_image(data_path+img)

        robo_referee.apply_homography()
        # robo_referee.get_BEV_transform()
        robo_referee.crop_img()
        robo_referee.contour_detection()
        robo_referee.ball_detection()

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

        cv.imshow("Frame", robo_referee.src)
        cv.waitKey()


main()