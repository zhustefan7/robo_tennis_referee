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

class Robo_Referee(object):
    def __init__(self,court_w = 540, court_h = 780):
        self.src = None
        self.margin = 50
        self.line_contour = None
        self.ball_loc = None
        self.H = np.array([[-3.97237654e-01 ,-1.35762559e+00 , 6.34568817e+02],
                            [-6.04181600e-16, -4.26827321e+00  ,1.46828599e+03],
                            [ 2.34038280e-05 ,-3.67005039e-03  ,1.00000000e+00]]
                            )
        self.min_x_1 = None
        self.min_y_1 = None
        self.max_x_2 = None
        self.max_y_2 = None

        #ball detection HSV thresh
        self.greenLower_front = (0, 69, 0)
        self.greenUpper_front = (79, 255, 255)

        self.greenLower_side = (0, 64, 142)
        self.greenUpper_side = (95, 255, 255)

        #for side view
        self.fps = 60
        self.vel_scale = 10
        self.contact_loc = None
        self.side_ball_loc = None
        self.side_ball_loc_prev = None

    def get_image(self, img_dir, side_img_dir):
        self.src = cv.imread(img_dir)       #modified by pipeline
        self.orig = cv.imread(img_dir)
        self.side_img = cv.imread(side_img_dir)
        self.side_img = cv.rotate(self.side_img, cv.ROTATE_90_CLOCKWISE)
        self.side_img = cv.resize(self.side_img,(int(self.side_img.shape[1]/2),int(self.side_img.shape[0]/2)))
    
    def get_BEV_transform(self):
        pitch = 80 #73.34
        yaw = -50 #37
        roll = 50  #3.5

        pitch *= np.pi/180
        yaw *= np.pi/180
        roll *= np.pi/180

        Rx = np.matrix([
        [1, 0,0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
        ])

        Ry = np.matrix([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        Rz = np.matrix([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
        ])

        Rrpy = Rx @ Ry @ Rz

        n = -np.array([[-0.08569279 , 0.97229642 , 0.21747749]]).reshape((-1,1))
        
        plane_eqn = [-0.01750559,  0.95802122,  0.28616261,  1.69369471]
        plane_center = np.array([ -0.55109537,   1.42110538, -10.71038818]).reshape((-1,1))




        ##720 HD
        fx = 692.27880859375
        fy = 692.27880859375
        cx = 684.1553344726562
        cy = 382.753662109375


        ##1080 HD
        # fx = 1427.2435302734375
        # fy = 1427.2435302734375
        # cx = 1037.1259765625
        # cy = 584.6063842773438

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])

        a = np.array([0,1,0]).reshape((-1,1))
        dot_prod = np.dot(a.flatten(), n.flatten())
        cross_prod_norm = np.linalg.norm(np.cross(a.flatten(),n.flatten()))
        G = np.array(
            [[dot_prod, -cross_prod_norm, 0],
            [cross_prod_norm, dot_prod, 0],
            [0, 0, 1]])

        u = a
        v = ((n-dot_prod * a)/np.linalg.norm(n-dot_prod * a)).reshape((-1,1))
        w = np.cross(n.flatten(),a.flatten()).reshape((-1,1))
        F = np.concatenate([u,v,w],axis=1)
        F = np.linalg.inv(F)

        U = np.linalg.inv(F) @ G @ F
        R = U

        d = plane_eqn[3]
        t = plane_center        #incorrect
        # print("n",n)
        # print("d",d)
        # print("t",t)
        # n = np.array([0,1,0]).reshape((-1,1))
        t = np.array([-10,-10, 0]).reshape((-1,1))        #increase y component: move front, increase z: move higher,
        # t = np.array([15, 20, 10]).reshape((-1,1))        #increase y component: move front, increase z: move higher,
        H = np.eye(3) + np.matmul(t,n.T)/d
        H = Rx @ Rz @ H
        H = np.matmul(np.matmul(K,H), np.linalg.inv(K))
        # print(H)

        dx = 0
        dy = 1200
        trans = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]],
            dtype = "float32")
        
        self.H = trans @ H

        self.src = cv2.warpPerspective(self.src, trans @ H,((3000,2000)))
        self.src = cv2.resize(self.src,(int(self.src.shape[1]/2),int(self.src.shape[0]/2)))

        # cv2.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png", self.src)


        # cv.imshow("Frame", self.src)
        # cv.waitKey()


    def apply_homography(self):
        self.src = cv2.warpPerspective(self.src, self.H,((3000,2000)))
        self.src = cv2.resize(self.src,(int(self.src.shape[1]/2),int(self.src.shape[0]/2)))
        # cv.imshow("Frame", self.src)
        # cv.waitKey()

    def crop_img(self):
        outer_contour = self.contour_detection()
        x_array = []
        y_array = []
        for item in outer_contour:
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


        if self.min_x_1 == None:
            self.min_x_1 ,self.min_y_1 = x_array[min_idx], y_array[min_idx]
            # max_x ,max_y = x_array[max_idx], y_array[max_idx]
            self.max_y_1 = max(y_array)

        #     cv.circle(self.src, (int(min_x), int(min_y)), int(5),
        # (0, 255, 255), 2)
        #     cv.circle(self.src, (int(max_x), int(max_y)), int(5),
        # (0, 255, 255), 2)

            self.src = self.src[:self.max_y_1,self.min_x_1:]

            # cv.imshow("Frame", self.src)
            # cv.waitKey()
            # cv2.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png", self.src)

            self.max_x_2, self.max_y_2 = detect_lines(self.src)


            self.src = self.src[:int(self.max_y_2)+self.margin,:int(self.max_x_2)+self.margin+20]
        else:
            self.src = self.src[:self.max_y_1,self.min_x_1:]
            self.src = self.src[:int(self.max_y_2)+self.margin,:int(self.max_x_2)+self.margin+20]

        # cv.imshow("Frame", self.src)
        # cv.waitKey()



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
            # # Show in a window
            # cv.imshow('Contours', drawing)
            # cv.waitKey()
            # cv.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/presentation_imgs/11_2_1080HD/38_contours.png", drawing)

            return contours[max_contour_idx]

        src_gray = cv.cvtColor(self.src, cv.COLOR_BGR2GRAY)
        src_gray = cv.blur(src_gray, (3,3))
        # Create Window
        # source_window = 'Source'
        # cv.namedWindow(source_window)
        # cv.imshow(source_window, self.src)
        max_thresh = 255
        thresh = 74 # initial threshold
        # cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
        self.line_contour = thresh_callback(thresh)
        return thresh_callback(thresh)
    
    def ball_detection(self,img,ball_loc,thresh_low,thresh_high):
        
        blurred = cv.GaussianBlur(img, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
        # print(hsv.shape)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv.inRange(hsv, thresh_low, thresh_high)
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
                # cv.circle(self.src, (int(x), int(y)), int(radius),
                #     (0, 0, 255), 2)
                # cv.circle(self.src, center, 5, (0, 0, 255), -1)
                ball_loc = center
            else:
                ball_loc = None
        return ball_loc
        # cv.imshow("Frame", mask)
        # cv.imshow("Frame", self.src)
        # cv.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/presentation_imgs/11_2_1080HD/38_warped.png", self.src)
        # cv.waitKey()
        # cv.destroyAllWindows()
    


    def line_judge(self):
        ball_x ,ball_y = self.ball_loc[0], self.ball_loc[1]
        if ball_x > self.min_x_1  and ball_x < self.max_x_2 and ball_y > self.min_y_1 and ball_y<self.max_y_2:
            return True
        return False


    def velocity_calc(self, img, ball_loc, ball_loc_prev, thresh_low, thresh_high):
        ball_loc = self.ball_detection(img, ball_loc, thresh_low, thresh_high)
        velocity = 0
        
        #calculate velocity vector and plot
        if ball_loc != None and ball_loc_prev != None:
            velocity_vec = tuple(map(lambda i, j: (i - j), ball_loc, ball_loc_prev))
            # print(velocity_vec)
            velocity = np.linalg.norm(np.array(velocity_vec)/(1/self.fps))
            print("velocity:",velocity)
            img = cv.arrowedLine(
                img,
                pt1=ball_loc,
                pt2=tuple(map(lambda i, j: i + j*self.vel_scale, ball_loc, velocity_vec)),
                color=(255,0,0), thickness=3) 

            # frame when ball touches ground
            if velocity_vec[0] == 0:
                velocity_slope = np.inf
            else:
                velocity_slope = velocity_vec[1]/velocity_vec[0]
            # print(velocity_slope)
            if self.contact_loc == None and velocity_slope >= 0:
                self.contact_loc = ball_loc_prev
            if self.contact_loc != None:
                cv.circle(img, self.contact_loc, 10, (0, 0, 255),-1)
            # print(self.contact_loc)

        ball_loc_prev = ball_loc
        return img, ball_loc, ball_loc_prev, velocity



        
# img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/HD720_SN14932_16-42-54/left000459.png'
# robo_referee =Robo_Referee(img_dir)
# robo_referee.apply_homography()
# # robo_referee.get_BEV_transform()
# robo_referee.crop_img()
# robo_referee.contour_detection()
# robo_referee.ball_detection()

# if robo_referee.ball_loc != None:
#     ball_in = robo_referee.line_judge()
#     x , y = robo_referee.ball_loc[0], robo_referee.ball_loc[1]
#     if ball_in:
#         cv.circle(robo_referee.src, (int(x), int(y)), int(10),
#         (0, 0, 255), 2)
#         print("Ball is in!!")
#     else:
#         cv.circle(robo_referee.src, (int(x), int(y)), int(10),
#         (0, 255, 255), 2)
#         print("Ball is out!")
# else:
#     print("No ball is detected")

# cv.imshow("Frame", robo_referee.src)
# cv.waitKey()



# if __name__ == "__main__":
#     img_dir = 