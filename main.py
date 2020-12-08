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
    data_path = "/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/HD720_SN14932_16-42-54_sync/"
    # data_path = "/home/hcl/Documents/ZED/12-2020videos/HD720_SN14932_16-42-54_sync/"
    side_img_path = "/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/side_img/"
    files = os.listdir(data_path)
    imgs =[]

    for file in files:
        if file.endswith('.png'):
            imgs.append(file)
    imgs = sorted(imgs)



    side_files = os.listdir(side_img_path)
    side_imgs =[]
    
    for file in side_files:
        if file.endswith('.png'):
            side_imgs.append(file)
    side_imgs = sorted(side_imgs)

    # Need to Change this number for different classes
    class_label =0
    frame_num = 1

    robo_referee =Robo_Referee()
    for img,side_img in zip(imgs,side_imgs):
        print("frame",frame_num)
        frame_num+=1
        robo_referee.get_image(data_path+img,side_img_path+side_img)

        #side view
        robo_referee.side_img, robo_referee.side_ball_loc, robo_referee.side_ball_loc_prev, robo_referee.side_velocity = robo_referee.velocity_calc(
            robo_referee.side_img, robo_referee.side_ball_loc, robo_referee.side_ball_loc_prev, robo_referee.greenLower_side, robo_referee.greenUpper_side
            )

        if robo_referee.side_ball_loc != None:
            cv.circle(robo_referee.side_img, (int(robo_referee.side_ball_loc[0]), int(robo_referee.side_ball_loc[1])), int(10),
                            (0, 255, 0), 2)

                        

        #front view
        robo_referee.apply_homography()
        # robo_referee.get_BEV_transform()
        robo_referee.crop_img()
        # robo_referee.contour_detection()
        ball_loc = robo_referee.ball_detection(robo_referee.orig,robo_referee.ball_loc,robo_referee.greenLower_front,robo_referee.greenUpper_front)
        if ball_loc != None:
            ball_loc_hom = np.array([[ball_loc[0]],[ball_loc[1]],[1]])
            ball_loc_hom = robo_referee.H@ball_loc_hom
            ball_loc_hom = ball_loc_hom/ball_loc_hom[2]
            robo_referee.ball_loc = (ball_loc_hom[0]/2-robo_referee.min_x_1, ball_loc_hom[1]/2)


            x , y = robo_referee.ball_loc[0], robo_referee.ball_loc[1]
            cv.circle(robo_referee.src, (int(x), int(y)), int(10),
                        (0, 255, 0), 2)
            cv.circle(robo_referee.orig, (int(ball_loc[0]), int(ball_loc[1])), int(10),
                        (0, 255, 0), 2)

                        

        if robo_referee.contact_loc != None:

            if robo_referee.front_contact_loc == None:

                robo_referee.front_contact_loc = (ball_loc[0],ball_loc[1])
                robo_referee.top_contact_loc = ( robo_referee.ball_loc[0], robo_referee.ball_loc[1])

                ball_in = robo_referee.line_judge()
            
            cv.circle(robo_referee.orig, robo_referee.front_contact_loc, 10, (0, 0, 255),-1)
            cv.circle(robo_referee.src, robo_referee.top_contact_loc, 10, (0, 0, 255),-1)

            if ball_in:
                # cv.circle(robo_referee.src, (int(x), int(y)), int(10),
                # (0, 255, 0), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                org = (100, 50) 
                
                # fontScale 
                fontScale = 1.5
                
                # Blue color in BGR 
                color = (0, 255, 0) 
                
                # Line thickness of 2 px 
                thickness = 3
                
                # Using cv2.putText() method 
                robo_referee.src = cv2.putText(robo_referee.src, 'In', org, font,  
                                fontScale, color, thickness, cv2.LINE_AA) 

                print("Ball is in!!")
            else:
                #     cv.circle(robo_referee.src, (int(x), int(y)), int(10),
            #     (0, 0, 255), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX 
                # org 
                org = (100, 50) 
                
                # fontScale 
                fontScale = 1.5
                
                # Blue color in BGR 
                color = (0, 0, 255) 
                
                # Line thickness of 2 px 
                thickness = 3
                
                # Using cv2.putText() method 
                robo_referee.src = cv2.putText(robo_referee.src, 'Out', org, font,  
                                fontScale, color, thickness, cv2.LINE_AA) 
                print("Ball is out!")

        
        
        if cv2.waitKey() == ord('q'):
            break

        H = robo_referee.side_img.shape[0]
        robo_referee.src = cv.resize(robo_referee.src,(int(H/robo_referee.src.shape[0]*robo_referee.src.shape[1]),H))
        robo_referee.orig = cv.resize(robo_referee.orig,(int(H/robo_referee.orig.shape[0]*robo_referee.orig.shape[1]),H))


        
        cv.imshow("Orig Frame",robo_referee.orig)
        cv.imshow("Warped Frame", robo_referee.src)
        cv.imshow("Side Frame",robo_referee.side_img)
        cv.waitKey()
        

main()