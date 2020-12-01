import numpy as np
import cv2

#load image
path = "/home/hcl/Documents/ZED/pics/Explorer_HD1080_SN14932_16-24-06.png"
im = cv2.imread(path)
h, w = im.shape[0], im.shape[1]
print("orignal image shape", h,w)
im = im[:,:int(w/2),:]      #get left image
im = cv2.resize(im,(int(im.shape[1]/2),int(im.shape[0]/2)))
cv2.imshow("im",im)
cv2.waitKey()
cv2.destroyAllWindows()

#ground plane
a = np.array([0,0,1])
n = np.array([-0.02410229,  0.95901954,  0.28231293])
plane = [-0.02410229,  0.95901954,  0.28231293,  1.65937257]
pitch = np.arcsin(-n[1])
yaw = np.arctan2(n[0], n[2])
print(pitch,yaw)

Rz = np.matrix([
[np.cos(yaw), -np.sin(yaw), 0],
[np.sin(yaw), np.cos(yaw), 0],
[0, 0, 1]
])

Ry = np.matrix([
[np.cos(pitch), 0, np.sin(pitch)],
[0, 1, 0],
[-np.sin(pitch), 0, np.cos(pitch)]
])


#intrinsics
fx = 699.04
fy = 699.04
cx = 678.56
cy = 357.492
k1 = -0.1719
k2 = 0.0245
p1 = 0
p2 = 0
k3 = 0

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]])

#warp with https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
if False:
    dot_prod = np.dot(a, n.T)
    cross_prod_norm = np.linalg.norm(np.cross(a,n))
    G = np.array(
        [[dot_prod, -cross_prod_norm, 0],
        [cross_prod_norm, dot_prod, 0],
        [0, 0, 1]])

    u = a.reshape((-1,1))
    v = ((n-dot_prod * a)/np.linalg.norm(n-dot_prod * a)).reshape((-1,1))
    w = np.cross(n,a).reshape((-1,1))

    F = np.concatenate([u,v,w],axis=1)
    F = np.linalg.inv(F)

    U = np.linalg.inv(F) @ G @ F
    # U = np.vstack((U,np.zeros(3)))
    # U = np.hstack((U,np.array([0,0,0,1]).reshape((-1,1))))
    print(U)
    M = K @ U

#Warp by obtaining homography
if True:
    # dx = 1
    # dy = 1
    # t = np.array([
    #     [1,0,dx],
    #     [0,1,dy],
    #     [0,0,1]])
    # print(t)

    #corners of original image
    img_corners = np.array([
        [160,300],
        [95,155],
        [525,112],
        [913,165]],
        dtype = "float32"
    )
    # known court length
    court_w = 200
    court_h = 200
    bird_eye_corners = np.array([
        [0,court_h],
        [0,0],
        [court_w, 0],
        [court_w,court_h]],
        dtype = "float32"
    )
    
    M = cv2.getPerspectiveTransform(img_corners, bird_eye_corners)
    print(M)

warp = cv2.warpPerspective(im, M ,((1000,1000)))

cv2.imshow("warp",warp)
cv2.waitKey()
cv2.destroyAllWindows()





