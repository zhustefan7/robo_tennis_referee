import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans



def calc_line_coeff(line):
    xa_1, ya_1,xa_2,ya_2 = line
    den = float((xa_2-xa_1))
    num = (ya_2-ya_1)
    M_a = num / den
    B_a = ya_1- M_a*xa_1 
    return np.array([-M_a, 1, -B_a])


# def cluster_lines(lines, all_lines_indices):
#     cluster_labels = np.zeros(len(all_lines_indices))
#     horizontal_slopes =[]
#     vertical_slopes =[]
#     horizontal_line_map = []
#     vertical_line_map = []

#     horizontal_clusters = []
#     vertical_clusters = []
#     horizontal_lines = np.array([])
#     vertical_lines = np.array([])

#     for i in all_lines_indices:
#         x1, y1, x2, y2 = lines[i][0]
#         line_coeff = calc_line_coeff(lines[i][0])
#         slope = -line_coeff[0]
#         if slope < 0:
#             horizontal_slopes.append([slope, math.sqrt(((y1+y2)/2)**2)])
#             horizontal_line_map.append(i)
#         else:
#             vertical_slopes.append([slope, math.sqrt(((x1+x2)/2)**2)])
#             vertical_line_map.append(i)
            
#     cluster_num = 3
#     h_kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(horizontal_slopes)
#     v_kmeans = KMeans(n_clusters=cluster_num, random_state=0).fit(vertical_slopes)

#     horizontal_line_map= np.array(horizontal_line_map)
#     vertical_line_map = np.array(vertical_line_map)
    

#     for i in range(cluster_num):
#         horizontal_clusters.append(horizontal_line_map[h_kmeans.labels_==i])
#         horizontal_lines= np.hstack((horizontal_lines, horizontal_line_map[h_kmeans.labels_==i]))
#     for i in range(cluster_num):
#         vertical_clusters.append(vertical_line_map[v_kmeans.labels_== int(i)])
#         vertical_lines= np.hstack((vertical_lines, vertical_line_map[v_kmeans.labels_==i]))
#     horizontal_clusters = np.array(horizontal_clusters)
#     vertical_clusters = np.array(vertical_clusters)
#     return horizontal_clusters, vertical_clusters, horizontal_lines, vertical_lines

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


def detect_line():
    # img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/video_frames/ezgif-frame-037.png'
    img_dir = '/home/stefanzhu/Documents/2020_Fall/16877_geo_vision/robo_referee/pics/warped.png'
    plt_img = plt.imread(img_dir)
    img = cv2.imread( img_dir,0)
    img = img[:,0:1920]
    # img = cv2.blur(img,(5,5))


    # Convert the image to gray-scale
    _,gray = cv2.threshold(img,100,255,cv2.THRESH_BINARY)

    # _,gray = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
    cv2.imshow('gray',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_filtered = cv2.bilateralFilter(gray, 7, 50, 50)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray_filtered,50, 500)

    # cv2.imshow('edges',edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 120, minLineLength=10, maxLineGap=10)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=250, maxLineGap=30)
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=150, maxLineGap=10)
    all_lines_indices = np.linspace(0,len(lines)-1,len(lines), dtype= int)
    horizontal_lines, vertical_lines = cluster_lines(lines,all_lines_indices)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        plt.plot([x1,x2],[y1,y2])

    # for i in range(len(cluster_labels)):
    #     if cluster_labels[i] == 2:
    #         x1, y1, x2, y2 = lines[i][0]
    #         plt.plot([x1,x2],[y1,y2])

    # for i in range(len(horizontal_clusters)):
    # for idx in vertical_clusters[0]:
    #     x1, y1, x2, y2 = lines[idx][0]
    #     plt.plot([x1,x2],[y1,y2])
    # intersection_pnts = []
    for i in range(len(horizontal_lines)):
        # intersection_count = 0
        for j in range(len(vertical_lines)):
            line1_indx = int(horizontal_lines[i])
            line2_indx = int(vertical_lines[j])
            line1_coeff = calc_line_coeff(lines[line1_indx][0])
            line2_coeff = calc_line_coeff(lines[line2_indx][0])
            intersect_p = np.cross(line1_coeff,line2_coeff)
            intersect_p = intersect_p / intersect_p[2]
            # plt.scatter([intersect_p[0]],[ intersect_p[1]])
            if intersect_p[2]!= 0 and intersect_pnt_on_both_lines(lines, line1_indx, line2_indx, intersect_p):
            #     intersection_count+=1
            # if intersection_count>=3:
            #     intersection_pnts.append(intersect_p)
                plt.scatter([intersect_p[0]],[ intersect_p[1]])


    plt.imshow(plt_img)

    min_iter = 0
    dist_thresh = 5
    satisfied_line_thresh = 3
    

    for i in range(min_iter):
        if len(all_lines_indices)!= 0:
            line_pair =  np.random.choice(all_lines_indices,2)
            line1_index, line2_index = line_pair[0], line_pair[1]

            if (lines_not_close(lines, line1_index, line2_index)):
                # print("line1 index",line1_index )
                # print("line2 index",line2_index )
                line1_coeff = calc_line_coeff(lines[line1_index][0])
                line2_coeff = calc_line_coeff(lines[line2_index][0])

                intersect_p = np.cross(line1_coeff,line2_coeff)
                if intersect_p[2]!= 0:
                    intersect_p = intersect_p / intersect_p[2]
                    satisfied_line_count = 0
                    satisfied_lines = []
                    for j in all_lines_indices:
                        if j!=line1_index and j!= line2_index:
                            dist = calc_pnt_to_line(lines[j][0], intersect_p)
                            if dist < dist_thresh:
                                satisfied_line_count +=1
                                satisfied_lines.append(j)
                    plt.scatter([intersect_p[0]],[ intersect_p[1]])
                    if satisfied_line_count > satisfied_line_thresh:
                        print(satisfied_lines)
                        # all_lines_indices = np.delete(all_lines_indices, np.array(satisfied_lines))
                        print(all_lines_indices)
                        plt.scatter([intersect_p[0]],[ intersect_p[1]])
    plt.show()



def intersect_pnt_on_both_lines(lines, line1_indx, line2_indx, intersect_p):
    xa_1, ya_1,xa_2,ya_2   = lines[line1_indx][0]
    xb_1, yb_1,xb_2,yb_2  = lines[line2_indx][0]
    margin = 1000

    if (intersect_p[0] > min(xa_1,xa_2)-margin) and (intersect_p[0] < max(xa_1,xa_2)+margin):
        if (intersect_p[1] > min(ya_1,ya_2)-margin) and (intersect_p[1] < max(ya_1,ya_2)+margin):
            if (intersect_p[0] > min(xb_1,xb_2)-margin) and (intersect_p[0] < max(xb_1,xb_2)+margin):
                if (intersect_p[1] > min(yb_1,yb_2)-margin) and (intersect_p[1] < max(yb_1,yb_2)+margin):

                    return True
    return False


def calc_pnt_to_line(line, vanish_pnt):
    line_coeff = calc_line_coeff(line)
    a,b,c = line_coeff[0], line_coeff[1], line_coeff[2]
    x0, y0 = vanish_pnt[0],vanish_pnt[1]
    d = abs(a*x0 + b*y0 +c)/math.sqrt(a**2 + b**2)
    return d

def lines_not_close(lines, line1_index, line2_index):
    line1  = lines[line1_index][0]
    xa_1, ya_1,xa_2,ya_2  = lines[line2_index][0]
    dist1= calc_pnt_to_line(line1, np.array([xa_1, ya_1,1]))
    dist2= calc_pnt_to_line(line1, np.array([xa_2, ya_2,1]))
    dist = math.sqrt(dist1**2 + dist2**2)
    # print(dist)
    if dist < 10:
        return False
    else:
        return True



if __name__ == "__main__":
    detect_line()