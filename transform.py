import numpy as np
import cv2

#load image
# path = "/home/hcl/Documents/ZED/pics/Explorer_HD1080_SN14932_16-24-06.png"
# path = "/home/hcl/Documents/ZED/pics/Explorer_HD720_SN14932_16-06-34.png"
path = "/home/hcl/Downloads/ezgif-4-c946bbd8d6fe-png-split/ezgif-frame-036.png"
im = cv2.imread(path)
h, w = im.shape[0], im.shape[1]
print("orignal image shape", h,w)
# im = im[:,:int(w/2),:]      #get left image
# im = cv2.resize(im,(int(im.shape[1]/4),int(im.shape[0]/4)))

cv2.imshow("im",im)
cv2.waitKey()
cv2.destroyAllWindows()

# intrinsics
# HD1080_SN14932_16-33-03
fx = 1427.2435302734375
fy = 1427.2435302734375
cx = 1037.1259765625
cy = 584.6063842773438

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]])

#ground plane
normal = [-0.01750558,  0.95802116,  0.28616259]
plane_eqn = [-0.01750559,  0.95802122,  0.28616261,  1.69369471]
plane_center = [ -0.55109537,   1.42110538, -10.71038818]

# RPY
if False:
    pitch = np.arcsin(-n[1])
    yaw = np.arctan2(n[0], n[2])
    print("pitch",pitch*180/np.pi)
    print("yaw",yaw*180/np.pi)


    dx = 0
    dy = 0

    roll = -70
    pitch = 0.1
    yaw = 0       #rotate image clockwise  Rz axis is into the image

    roll *= np.pi/180
    pitch *= np.pi/180
    yaw *= np.pi/180

    Rx = np.matrix([
    [1, 0,0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
    ])

    Ry = np.matrix([
    [np.cos(pitch), 0, np.sin(pitch)],
    [0, 1, 0],
    [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.matrix([
    [np.cos(yaw), -np.sin(yaw), 0],
    [np.sin(yaw), np.cos(yaw), 0],
    [0, 0, 1]
    ])

    M = Rx

#Dyadic prod & Axis Angle
if False:
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
    # M = K @ U
    M = U

#Warp by obtaining homography
if True:
    #corners of original image
    # img_corners = np.array([
    #     [160,300],
    #     [95,155],
    #     [525,112],
    #     [913,165]],
    #     dtype = "float32"
    # )
    #corners of original image
    img_corners = np.array([
        [172,142],
        [75,75],
        [245,63],
        [460,86]],
        dtype = "float32"
    )
    img_corners *= 4

    # known court length
    court_w = 27 * 10
    court_h = (21+18)*10
    bird_eye_corners = np.array([
        [0,court_h],
        [0,0],
        [court_w, 0],
        [court_w,court_h]],
        dtype = "float32"
    )
    
    M = cv2.getPerspectiveTransform(img_corners, bird_eye_corners)
    print(M)

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3355419/
if False:
    f = fx
    d = plane[3]
    roll = -73.54*np.pi/180  # +/-???
    print("distance to ground",d)
    print("focal length",f)
    print(roll)

    # im = cv2.resize(im,(int(im.shape[1]/2),int(im.shape[0]/2)))
    h=im.shape[0]
    w=im.shape[1]
    print(w,h)
    warp = np.zeros((1000,1000,3))

    cv2.imshow("im",im[280:600,177:1000,:])
    cv2.waitKey()
    cv2.destroyAllWindows()
    for x in range(177,1000):
        for y in range(280,600):
            shift = np.abs(d*(np.sin(roll)+f*np.cos(roll)) / (f*np.sin(roll)-np.cos(roll)))+1
            x_w = d*(x*np.sin(roll)+f*np.cos(roll)) / (-y*np.cos(roll)+f*np.sin(roll)) + shift
            y_w = d*(y*np.sin(roll)+f*np.cos(roll))/(-y*np.cos(roll)+f*np.sin(roll)) + shift
            intensity = im[y,x,:]/255
            # x_w = x_w*100
            # y_w = y_w*100
            print("xy",x,y)
            print("new xy", x_w,y_w)
            x_w = int(x_w)
            y_w = int(y_w)
            print(intensity)
            warp[y_w,x_w,:] = intensity

#homography equation
if False:
    # H = Rx - np.matmul(t,n.T)/d
    H = np.matmul(np.matmul(K,Rx), np.linalg.inv(K))


#warp
dx = 0
dy = 0
trans = np.array([
    [1, 0, dx],
    [0, 1, dy],
    [0, 0, 1]],
    dtype = "float32")

warp = cv2.warpPerspective(im, trans @ M ,((1000,1000)))
# warp = cv2.resize(warp,(int(warp.shape[1]/2),int(warp.shape[0]/2)))

cv2.imshow("warp",warp)
cv2.waitKey()
cv2.destroyAllWindows()