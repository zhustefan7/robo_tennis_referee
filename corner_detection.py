from __future__ import print_function
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2 
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') 
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans






def intersect_pnt_on_both_lines(lines, line1_indx, line2_indx, intersect_p):
    xa_1, ya_1,xa_2,ya_2   = lines[line1_indx][0]
    xb_1, yb_1,xb_2,yb_2  = lines[line2_indx][0]
    margin = 100

    if (intersect_p[0] > min(xa_1,xa_2)-margin) and (intersect_p[0] < max(xa_1,xa_2)+margin):
        if (intersect_p[1] > min(ya_1,ya_2)-margin) and (intersect_p[1] < max(ya_1,ya_2)+margin):
            if (intersect_p[0] > min(xb_1,xb_2)-margin) and (intersect_p[0] < max(xb_1,xb_2)+margin):
                if (intersect_p[1] > min(yb_1,yb_2)-margin) and (intersect_p[1] < max(yb_1,yb_2)+margin):

                    return True
    return False



def calc_line_coeff(line):
    xa_1, ya_1,xa_2,ya_2 = line
    den = float((xa_2-xa_1))
    num = (ya_2-ya_1)
    M_a = num / den
    B_a = ya_1- M_a*xa_1 
    return np.array([-M_a, 1, -B_a])


def cluster_lines(lines, all_lines_indices):
    cluster_labels = np.zeros(len(all_lines_indices))
    horizontal_lines = []
    vertical_lines = []

    for i in all_lines_indices:
        x1, y1, x2, y2 = lines[i][0]
        line_coeff = calc_line_coeff(lines[i][0])
        slope = -line_coeff[0]
        if slope < 0:
            horizontal_lines.append(i)
        else:
            vertical_lines.append(i)
            
    return horizontal_lines, vertical_lines


def detect_corners(img):
        
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray,140,255,cv2.THRESH_BINARY)
    # img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
    # plt_img = plt.imread(img_dir)

    # _,gray = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray_filtered,50, 500)

    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=30, maxLineGap=10)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=250, maxLineGap=30)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=150, maxLineGap=10)
    all_lines_indices = np.linspace(0,len(lines)-1,len(lines), dtype= int)
    horizontal_lines, vertical_lines = cluster_lines(lines,all_lines_indices)

    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     plt.plot([x1,x2],[y1,y2])

    # for i in range(len(cluster_labels)):
    #     if cluster_labels[i] == 2:
    #         x1, y1, x2, y2 = lines[i][0]
    #         plt.plot([x1,x2],[y1,y2])

    # for i in range(len(horizontal_clusters)):
    # for idx in vertical_clusters[0]:
    #     x1, y1, x2, y2 = lines[idx][0]
    #     plt.plot([x1,x2],[y1,y2])
    # intersection_pnts = []

    corners = []
    
    for i in range(len(horizontal_lines)):
        # intersection_count = 0
        for j in range(len(vertical_lines)):
            line1_indx = int(horizontal_lines[i])
            line2_indx = int(vertical_lines[j])
            line1_coeff = calc_line_coeff(lines[line1_indx][0])
            line2_coeff = calc_line_coeff(lines[line2_indx][0])
            intersect_p = np.cross(line1_coeff,line2_coeff)

            if intersect_p[2]!= 0 and intersect_pnt_on_both_lines(lines, line1_indx, line2_indx, intersect_p):
                intersect_p = intersect_p / intersect_p[2]
            #     intersection_count+=1
            # if intersection_count>=3:
            #     intersection_pnts.append(intersect_p)
                corners.append([intersect_p[0],intersect_p[1]])
                plt.scatter([intersect_p[0]],[ intersect_p[1]])
    # plt.imshow(plt_img)
    # plt.show()
    return corners

def detect_lines(img):
        
    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,gray = cv2.threshold(gray,140,255,cv2.THRESH_BINARY)
    # img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
    # plt_img = plt.imread(img_dir)

    # _,gray = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    # cv2.imshow('gray',gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray_filtered,50, 500)

    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=30, maxLineGap=10)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=250, maxLineGap=30)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=150, maxLineGap=10)
    all_lines_indices = np.linspace(0,len(lines)-1,len(lines), dtype= int)
    horizontal_lines, vertical_lines = cluster_lines(lines,all_lines_indices)

    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     plt.plot([x1,x2],[y1,y2])

    # for i in range(len(cluster_labels)):
    #     if cluster_labels[i] == 2:
    #         x1, y1, x2, y2 = lines[i][0]
    #         plt.plot([x1,x2],[y1,y2])

    # for i in range(len(horizontal_clusters)):
    # for idx in vertical_clusters[0]:
    #     x1, y1, x2, y2 = lines[idx][0]
    #     plt.plot([x1,x2],[y1,y2])
    # intersection_pnts = []

    corners = []

    x_array = []
    y_array = []

    for i in range(len(horizontal_lines)):
        line_idx = int(horizontal_lines[i])
        x1, y1, x2, y2 = lines[line_idx][0]
        y_array.append(max(y1,y2))

    for i in range(len(vertical_lines)):
        line_idx = int(vertical_lines[i])
        x1, y1, x2, y2 = lines[line_idx][0]
        x_array.append(max(x1,x2))
    # print(max(x_array))
    # print(max(y_array))

    return (max(x_array),max(y_array) )

    
    # for i in range(len(horizontal_lines)):
    #     # intersection_count = 0
    #     for j in range(len(vertical_lines)):
    #         line1_indx = int(horizontal_lines[i])
    #         line2_indx = int(vertical_lines[j])
    #         line1_coeff = calc_line_coeff(lines[line1_indx][0])
    #         line2_coeff = calc_line_coeff(lines[line2_indx][0])
    #         intersect_p = np.cross(line1_coeff,line2_coeff)

    #         if intersect_p[2]!= 0 and intersect_pnt_on_both_lines(lines, line1_indx, line2_indx, intersect_p):
    #             intersect_p = intersect_p / intersect_p[2]
    #         #     intersection_count+=1
    #         # if intersection_count>=3:
    #         #     intersection_pnts.append(intersect_p)
    #             corners.append([intersect_p[0],intersect_p[1]])
    #             plt.scatter([intersect_p[0]],[ intersect_p[1]])
    # # plt.imshow(plt_img)
    # # plt.show()
    # return corners


# img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
# src = cv2.imread(img_dir)
# detect_lines(src)
# # detect_corners(src)