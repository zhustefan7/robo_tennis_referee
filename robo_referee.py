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

class Robo_Referee(object):
    def __init__(self,img_dir,court_w = 540, court_h = 780):
        self.src = cv.imread(img_dir)
        self.margin = 20
        self.line_contour = None
        self.ball_loc = None
        self.H = np.eye(3)

    

    def get_BEV_transform(self):
        pitch = 30 #73.34
        yaw = -50 #37
        roll = 42  #3.5

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

        n = np.array([-0.01750558, 0.95802116, 0.28616259]).reshape((-1,1))
        plane_eqn = [-0.01750559,  0.95802122,  0.28616261,  1.69369471]
        plane_center = np.array([ -0.55109537,   1.42110538, -10.71038818]).reshape((-1,1))


        fx = 1427.2435302734375
        fy = 1427.2435302734375
        cx = 1037.1259765625
        cy = 584.6063842773438

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
        t = np.array([0, 20, 20]).reshape((-1,1))        #increase y component: move front, increase z: move higher,
        # t = np.array([15, 20, 10]).reshape((-1,1))        #increase y component: move front, increase z: move higher,
        H = R + np.matmul(t,n.T)/d
        H = Rx @ Rz @ H
        H = np.matmul(np.matmul(K,H), np.linalg.inv(K))
        # print(H)

        dx = 0
        dy = 500
        trans = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]],
            dtype = "float32")
        
        self.H = trans @ H

        self.src = cv2.warpPerspective(self.src, trans @ H,((3000,2000)))
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


        min_x ,min_y = x_array[min_idx], y_array[min_idx]
        # max_x ,max_y = x_array[max_idx], y_array[max_idx]

    #     cv.circle(self.src, (int(min_x), int(min_y)), int(5),
    # (0, 255, 255), 2)
    #     cv.circle(self.src, (int(max_x), int(max_y)), int(5),
    # (0, 255, 255), 2)

        self.src = self.src[:,min_x:]

        corners = detect_corners(self.src)
        max_idx =0
        max_sum =0
        for i in range(len(corners)):
            x,y = corners[i][0],corners[i][1]
            sum = x+y
            if sum > max_sum:
                max_sum = sum
                max_idx = i
            



        max_x, max_y = corners[max_idx][0],corners[max_idx][1]

        # max_y_idx = np.argmax(corners[:,1])

        # print(max_y_idx)
    #     for corner in corners:
    #         x,y = corner[0],corner[1]
    #         cv.circle(self.src, (int(x), int(y)), int(5),
    # (0, 0, 255), 2)



    #     cv.imshow("Frame", self.src)
    #     cv.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/presentation_imgs/11_2_1080HD/38_corners.png", self.src)

    #     cv.waitKey()

        self.src = self.src[:int(max_y)+self.margin,:int(max_x)+self.margin]



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
        cv.waitKey()
        return thresh_callback(thresh)
    
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
                # cv.circle(self.src, (int(x), int(y)), int(radius),
                #     (0, 0, 255), 2)
                # cv.circle(self.src, center, 5, (0, 0, 255), -1)
                self.ball_loc = center
        # cv.imshow("Frame", mask)
        # cv.imshow("Frame", self.src)
        # cv.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/presentation_imgs/11_2_1080HD/38_warped.png", self.src)
        # cv.waitKey()
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





        
img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/video_frames/ezgif-frame-038.png'
robo_referee =Robo_Referee(img_dir)
robo_referee.get_BEV_transform()
robo_referee.crop_img()
robo_referee.contour_detection()
robo_referee.ball_detection()

if robo_referee.ball_loc != None:
    ball_in = robo_referee.line_judge()
    x , y = robo_referee.ball_loc[0], robo_referee.ball_loc[1]
    if ball_in:
        cv.circle(robo_referee.src, (int(x), int(y)), int(10),
        (0, 0, 255), 2)
        print("Ball is in!!")
    else:
        cv.circle(robo_referee.src, (int(x), int(y)), int(10),
        (0, 255, 255), 2)
        print("Ball is out!")
else:
    print("No ball is detected")

cv.imshow("Frame", robo_referee.src)
cv.waitKey()



# if __name__ == "__main__":
#     img_dir = 