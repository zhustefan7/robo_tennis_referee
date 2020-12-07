import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv under python3
import cv2 
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import numpy as np
 
def nothing(x):
    pass
 
# Open the camera
cap = cv2.VideoCapture(0) 
 
# Create a window
cv2.namedWindow('image')
 
# create trackbars for color change
cv2.createTrackbar('lowH','image',0,179,nothing)
cv2.createTrackbar('highH','image',179,179,nothing)
 
cv2.createTrackbar('lowS','image',0,255,nothing)
cv2.createTrackbar('highS','image',255,255,nothing)
 
cv2.createTrackbar('lowV','image',0,255,nothing)
cv2.createTrackbar('highV','image',255,255,nothing)
 
# img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/HD720_SN14932_16-42-54/left000459.png'
# img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
img_dir = "/home/hcl/Desktop/GeoVis_Project_Tennis_Tracker/side_img/20201206_164251_375.png"

video_path = "/home/hcl/Documents/ZED/11-2020_videos/HD1080_SN14932_16-31-32.avi"
frame_num = 145     #frame of video to tune
use_vid = False

if use_vid:
    cap = cv2.VideoCapture(video_path)
    i = 1
    ret = True
    while ret == True:
        ret, vid_frame = cap.read()
        cv2.imshow('image', vid_frame)
        cv2.waitKey(1)
        print(i)
        if i==frame_num:
            break
        i += 1

while True:
    frame = cv2.imread(img_dir)

    if use_vid:
        frame = vid_frame

    h, w = frame.shape[0], frame.shape[1]
    # frame = frame[:,:int(w/2),:]    
    frame = cv2.resize(frame,(500,500))
 
    # get current positions of the trackbars
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    # convert color to hsv because it is easy to track colors in this color model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    print(lower_hsv)
    print(higher_hsv)
    # Apply the cv2.inrange method to create a mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    # Apply the mask on the image to extract the original color
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('image', frame)
    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()