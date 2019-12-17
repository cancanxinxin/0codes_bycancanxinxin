# coding:utf-8
import cv2
import numpy as np
import glob

# input: the pixel points data from the picture
# ouput: PnP algorithm output
# 参考网址：https://www.cnblogs.com/aoru45/p/9781540.html
# openCV官方文档：https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp
def PoseEstimate(ABCD):
       # Load previously saved data
       a = np.load('B.npy')
       # 白色标签尺寸为34*25，单位为mm,以C点位坐标原点
       # object_3d_points = np.array(([34,  0, 0],
       #                              [34, 24, 0],
       #                              [ 0, 24, 0],
       #                              [ 0,  0, 0]), dtype=np.double)	# ABCD in real world

       object_3d_points = np.array(([34, 24, 0],
                                    [0, 24, 0],
                                    [0, 0, 0],
                                    [34,  0, 0]), dtype=np.double)    # ABCD in real world

       # data 1219
       # [880, 435], [1132, 440], [1138, 617], [878, 622]
       object_2d_point = np.array(ABCD, dtype=np.double)	# ABCD in the image plane
       camera_matrix = np.array(a[0], dtype=np.double)
       dist_coefs = np.array(a[1], dtype=np.double)

       # to debug
       # print(object_3d_points)
       # print(object_2d_point)
       # print(camera_matrix) 
       # print(dist_coefs)

       # 求解相机位姿
       found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs,SOLVEPNP_DLS)
       rotM = cv2.Rodrigues(rvec)[0]
       camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
       print(camera_postion.T)
       print(found)
       print(rvec)
       print(tvec)

# the reason to use PnP is that we can measure the distance precisely
# to make the L operation
def PoseEstimate1(ABCD):
       # Load previously saved data
       a = np.load('B.npy')
       # 白色标签尺寸为34*25，单位为mm,以C点zuowei坐标原点
       object_3d_points = np.array(([34, 24, 0],
                                    [0, 24, 0],
                                    [0, 0, 0],
                                    [34,  0, 0]), dtype=np.double)    # ABCD in real world

       object_2d_point = np.array(ABCD, dtype=np.double)    # ABCD in the image plane
       camera_matrix = np.array(a[0], dtype=np.double)
       dist_coefs = np.array(a[1], dtype=np.double)

       # 求解相机位姿
       found, rvec, tvec = cv2.solvePnP(object_3d_points, object_2d_point, camera_matrix, dist_coefs)
       rotM = cv2.Rodrigues(rvec)[0]
       camera_postion = -np.matrix(rotM).T * np.matrix(tvec)
       # rvec = rvec * (180/3.1416)
       # print(camera_postion.T)
       # print(found)
       # print(rvec)
       print(tvec)
       # print(tvec[0])
       return tvec
