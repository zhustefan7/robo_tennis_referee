import numpy as np
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 

#load image
# path = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/Explorer_HD720_SN14932_16-11-09.png'
path = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/test.png'
im = cv2.imread(path)
h, w = im.shape[0], im.shape[1]
print("orignal image shape", h,w)
# im = im[:,:int(w/2),:]      #get left image
im = cv2.resize(im,(int(im.shape[1]/2),int(im.shape[0]/2)))
cv2.imshow("im",im)
cv2.waitKey()
cv2.destroyAllWindows()

#ground plane
a = np.array([0,0,1])
n = np.array([-0.02410229,  0.95901954,  0.28231293])
# n = np.array([0.2574, 3.44,0.274])
plane = [-0.02410229,  0.95901954,  0.28231293,  1.65937257]
pitch = np.arcsin(-n[1])
yaw = np.arctan2(n[0], n[2])
print(pitch,yaw)

yaw = -np.pi/16
Rz = np.matrix([
[np.cos(yaw), -np.sin(yaw), 0],
[np.sin(yaw), np.cos(yaw), 0],
[0, 0, 1]
])

# pitch = np.pi/180*10
pitch = 0
Ry = np.matrix([
[np.cos(pitch), 0, np.sin(pitch)],
[0, 1, 0],
[-np.sin(pitch), 0, np.cos(pitch)]
])

roll = np.pi/2
Rx = np.matrix([
[1, 0,0],
[0, np.cos(roll), -np.sin(roll)],
[0, np.sin(roll), np.cos(roll)]
]) 

print(Rx)

#intrinsics
# fx = 699.04
# fy = 699.04
# cx = 678.56
# cy = 357.492

fx = 1427.2435302734375
fy = 1427.2435302734375
cx = 1037.1259765625
cy = 584.6063842773438



k1 = -0.1719
k2 = 0.0245
p1 = 0
p2 = 0
k3 = 0

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]])


#####Dyadic Product
R_1 = 1/3*np.matmul(a[:,np.newaxis], np.array([[1/n[0], 1/n[1], 1/n[2]]])) 

#####Axis Angle
u = np.cross(a,n)/np.linalg.norm(np.cross(a,n))
u_hat = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
u = u[:,np.newaxis]

sin = np.linalg.norm(np.cross(a,n))/(np.linalg.norm(a)*np.linalg.norm(n))
cos = np.linalg.norm(np.dot(a,n))/(np.linalg.norm(a)*np.linalg.norm(n))
rot = np.eye(3) * cos + np.matmul(u,np.transpose(u))*(1-cos) + u_hat*sin

rot_inv = np.linalg.inv(rot) 
print("rot")
print(rot)
print(cv2.Rodrigues(rot))
transformed = np.matmul(rot,a)
print("transformed", transformed)
d = 1.65937257
# t= np.array([[1.01578724], [1.05884469], [-9.4285984]])
t= np.array([[0], [0], [0]])
# t= np.array([[1.01578724], [1.05884469], [-9.4285984]])
# t= np.array([[-10.0489], [-528.9226], [ 1.2236]])

n_tmp = n[np.newaxis,:]
print('t1',np.matmul(t,n_tmp))
# H = np.matmul(Rx,Ry)- np.matmul(t,n_tmp)/d
H = Rx- np.matmul(t,n_tmp)/d
H = np.matmul(np.matmul(K,H), np.linalg.inv(K))
translate = np.array([[1,0,500],[0,1,1800],[0,0,1]])
H = np.matmul(translate,H)
# H = np.matmul(Ry,H)
print("H", H.shape)
# print(np.linalg.norm(np.cross(a,transformed))/(np.linalg.norm(a)*np.linalg.norm(transformed)))


#Warp by obtaining homography
if True:
    dx = 0
    dy = 0
    t = np.array([
        [1,0,dx],
        [0,1,dy],
        [0,0,1]])
    # print(t)

    #corners of original image
    img_corners = np.array([
        [344,284],
        [150,153],
        [488,127],
        [915,177]],
        dtype = "float32"
    )
    # known court length
    scale = 20
    court_w = 27*scale
    court_h = 39*scale
    bird_eye_corners = np.array([
        [0,court_h],
        [0,0],
        [court_w, 0],
        [court_w,court_h]],
        dtype = "float32"
    )
    

    M = cv2.getPerspectiveTransform(img_corners, bird_eye_corners)
    M = np.matmul(t,M)
    print("M")
    print(M)
    # M = rot_inv - np.matmul(t,n_tmp)/d
    matmul_t_n_tmp = (rot_inv-M)*d
    print('matmul_t_n_tmp',matmul_t_n_tmp)




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
    # print("U")
    # print(U)
    M = K @ U
    

warp = cv2.warpPerspective(im, M,((1000,1000)))
# warp = cv2.resize(warp,(500,500))
cv2.imshow("warp",warp)
cv2.imwrite("/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png", warp)
cv2.waitKey()
cv2.destroyAllWindows()





