########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

# HD1080_SN14932_16-33-03
"""
    Plane sample that save the floor plane detected as a sl.
"""
import sys
import pyzed.sl as sl
from time import sleep

import numpy as np
import cv2

def main():
    cam = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.ULTRA
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Reading SVO file: {0}".format(filepath))
        init.set_from_svo_file(filepath)

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)

    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
    spatial = sl.SpatialMappingParameters()

    transform = sl.Transform()
    tracking = sl.PositionalTrackingParameters(transform)
    cam.enable_positional_tracking(tracking)

    pymesh = sl.Mesh()
    pyplane = sl.Plane()
    reset_tracking_floor_frame = sl.Transform()
    found = 0
    print("Processing...")
    i = 0




    calibration_params = cam.get_camera_information().calibration_parameters
    # Focal length of the left eye in pixels
    focal_left_x = calibration_params.left_cam.fx
    focal_left_y = calibration_params.left_cam.fy
    # First radial distortion coefficient
    # k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on z-axis
    # tz = calibration_params.T.z
    # Horizontal field of view of the left eye in degrees
    # h_fov = calibration_params.left_cam.h_fov
    print("fx", focal_left_x)
    print("fy", focal_left_y)
    print("cx", calibration_params.left_cam.cx)
    print("cy", calibration_params.left_cam.cy)

    

    cloud = sl.Mat()
    while i < 1000:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            err = cam.find_floor_plane(pyplane, reset_tracking_floor_frame)
            if err == sl.ERROR_CODE.SUCCESS:
            # if i > 200 and err == sl.ERROR_CODE.SUCCESS:
                found = 1
                print('Floor found!')
                pymesh = pyplane.extract_mesh()
                #get gound plane info
                print("floor normal",pyplane.get_normal())     #print normal of plane
                print("floor equation", pyplane.get_plane_equation())
                print("plane center", pyplane.get_center())
                print("plane pose", pyplane.get_pose())

                # camera_pose = sl.Pose()
                # py_translation = sl.Translation()
                # tracking_state = cam.get_position(camera_pose)
                # # if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # rotation = camera_pose.get_rotation_vector()
                # rx = round(rotation[0], 2)
                # ry = round(rotation[1], 2)
                # rz = round(rotation[2], 2)

                # translation = camera_pose.get_translation(py_translation)
                # tx = round(translation.get()[0], 2)
                # ty = round(translation.get()[1], 2)
                # tz = round(translation.get()[2], 2)

                # text_translation = str((tx, ty, tz))
                # text_rotation = str((rx, ry, rz))
                # pose_data = camera_pose.pose_data(sl.Transform())
                # print("translation",text_translation)
                # print("rotation",text_rotation)

                cam.retrieve_measure(cloud, sl.MEASURE.XYZRGBA)

                break

                
            i+=1

    if found == 0:
        print('Floor was not found, please try with another SVO.')
        cam.close()
        exit(0)

    #load image
    path = "/home/hcl/Documents/ZED/pics/Explorer_HD1080_SN14932_16-24-06.png"
    im = cv2.imread(path)
    h, w = im.shape[0], im.shape[1]
    print("orignal image shape", h,w)
    im = im[:,:int(w/2),:]      #get left image
    # im = cv2.resize(im,(int(im.shape[1]/2),int(im.shape[0]/2)))


    #warped img
    warped = np.zeros((h,w,3))

    roll = -80
    roll *= np.pi/180

    Rx = np.matrix([
    [1, 0,0],
    [0, np.cos(roll), -np.sin(roll)],
    [0, np.sin(roll), np.cos(roll)]
    ])

    #point cloud
    # i = 500
    # j = 355
    if False:
        point_cloud = sl.Mat()
        cam.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        for i in range(100):
            for j in range(1000):        
                point3D = point_cloud.get_value(i,j)
                # x = point3D[0]
                # y = point3D[1]
                # z = point3D[2]
                color = point3D[1][3]
                if ~np.isnan(color) and ~np.isinf(color):
                    xyz = point3D[1][:3].reshape((-1,1))
                    # xyz_hom = np.vstack((xyz,1))
                    print('point3D',point3D)
                    warped_index = Rx @ xyz             #NEED TO USE HOMO COORDINATES? divide by last column?
                    print('warped_index',warped_index)
                    RGB = im[j,i,:]
                    RGB = np.dstack(([255],[255],[255]))
                    print('RGB',RGB)
                    warped[int(np.round(warped_index[0]+10)),int(np.round(warped_index[1])),:] = RGB

    cam.disable_positional_tracking()
    # save_mesh(pymesh)
    cam.close()
    print("\nFINISH")

    # cv2.imshow('warp',warped)
    # cv2.waitKey()

    


def save_mesh(pymesh):
    while True:
        res = input("Do you want to save the mesh? [y/n]: ")
        if res == "y":
            msh = sl.ERROR_CODE.FAILURE
            while msh != sl.ERROR_CODE.SUCCESS:
                filepath = input("Enter filepath name: ")
                msh = pymesh.save(filepath)
                print("Saving mesh: {0}".format(repr(msh)))
                if msh:
                    break
                else:
                    print("Help : you must enter the filepath + filename without extension.")
            break
        elif res == "n":
            print("Mesh will not be saved.")
            break
        else:
            print("Error, please enter [y/n].")

if __name__ == "__main__":
    main()
