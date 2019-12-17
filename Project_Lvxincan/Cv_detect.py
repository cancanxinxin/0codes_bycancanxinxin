# Cv detect and localization
#coding=utf-8 
import cv2 as cv
import numpy as np
from matplotlib import pylab as plt
import argparse
import glob
from math import pi
from PoseEstimation12 import *
from RobotControl import *

# to find the known object and print the corner point location
def match(c):   
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='rec_small1.jpg')
    parser.add_argument('--input2', help='Path to input image 2.', default='frame' + str(c) + '.jpg')
    args = parser.parse_args()
    img_object = cv.imread(args.input1, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(args.input2, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    #-- Show detected matches
    cv.imshow('Good Matches & Object detection', img_matches)
    cv.imwrite('Good Matches&Object detection'+str(c) + '.jpg',img_matches)
    # print(int(scene_corners[0,0,0]),img_object.shape[1],int(scene_corners[0,0,1]))
    # print('Good Matches&Object detection1'+str(c) + '.jpg')
    
    # center_point0=(int(scene_corners[0,0,0] + img_object.shape[1])+int(scene_corners[2,0,0] + img_object.shape[1]))/2
    # center_point1=(int(scene_corners[0,0,1])+int(scene_corners[2,0,1]))/2
    # center_point=[center_point0,center_point1]
    # print(center_point)
    # return(center_point)

def match_once_printABCD():
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='rec_small1.jpg')
    parser.add_argument('--input2', help='Path to input image 2.', default='frame5.jpg')
    args = parser.parse_args()
    img_object = cv.imread(args.input1, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(args.input2, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    cv.circle(img_matches,(int(img_object.shape[0]),int(img_object.shape[1])),3,(0,255,0))
    #-- Show detected matches
    cv.imshow('Good Matches & Object detection', img_matches)
    cv.imwrite('Good Matches&Object detection1.jpg',img_matches)
    # shape[0]代表高度，shape[1]代表宽度
    print(img_object.shape[0],img_object.shape[1])

    #输出对应顺序为CDAB
    # print("A:",int(scene_corners[0,0,0]),',',int(scene_corners[0,0,1]))
    # print("B:",int(scene_corners[1,0,0]),',',int(scene_corners[1,0,1]))
    # print("C:",int(scene_corners[2,0,0]),',',int(scene_corners[2,0,1]))
    # print("D:",int(scene_corners[3,0,0]),',',int(scene_corners[3,0,1]))
    # ABCD = ([int(scene_corners[0,0,0]),int(scene_corners[0,0,1])],
    #         [int(scene_corners[1,0,0]),int(scene_corners[1,0,1])],
    #         [int(scene_corners[2,0,0]),int(scene_corners[2,0,1])],
    #         [int(scene_corners[3,0,0]),int(scene_corners[3,0,1])])

    ABCD = ([int(scene_corners[2,0,0]),int(scene_corners[2,0,1])],
            [int(scene_corners[3,0,0]),int(scene_corners[3,0,1])],
            [int(scene_corners[0,0,0]),int(scene_corners[0,0,1])],
            [int(scene_corners[1,0,0]),int(scene_corners[1,0,1])])    
    return(ABCD)
    cv.waitKey()    

def match_once_printABCD1(c):
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='rec_small1.jpg')
    parser.add_argument('--input2', help='Path to input image 2.', default='frame' + str(c) + '.jpg')
    args = parser.parse_args()
    img_object = cv.imread(args.input1, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(args.input2, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    cv.circle(img_matches,(int(img_object.shape[0]),int(img_object.shape[1])),3,(0,255,0))
    #-- Show detected matches
    cv.imshow('Good Matches & Object detection', img_matches)
    cv.imwrite('Good Matches&Object detection1.jpg',img_matches)
    # shape[0]代表高度，shape[1]代表宽度
    # print(img_object.shape[0],img_object.shape[1])

    #输出对应顺序为CDAB
    ABCD = ([int(scene_corners[2,0,0]),int(scene_corners[2,0,1])],
            [int(scene_corners[3,0,0]),int(scene_corners[3,0,1])],
            [int(scene_corners[0,0,0]),int(scene_corners[0,0,1])],
            [int(scene_corners[1,0,0]),int(scene_corners[1,0,1])])    
    return(ABCD)
    cv.waitKey()  

#   to be confirmed to be used well or not!!  Attention  !!
def match_printABCD(c):
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='rec_small1.jpg')
    parser.add_argument('--input2', help='Path to input image 2.', default='Good Matches&Object detection'+str(c) + '.jpg')
    args = parser.parse_args()
    img_object = cv.imread(args.input1, cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(args.input2, cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)
    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
    cv.circle(img_matches,(int(img_object.shape[0]),int(img_object.shape[1])),3,(0,255,0))
    #-- Show detected matches
    cv.imshow('Good Matches & Object detection', img_matches)
    cv.imwrite('Good Matches&Object detection1.jpg',img_matches)
    # shape[0]代表高度，shape[1]代表宽度
    print(img_object.shape[0],img_object.shape[1])

    #输出对应顺序为CDAB
    # print("A:",int(scene_corners[0,0,0]),',',int(scene_corners[0,0,1]))
    # print("B:",int(scene_corners[1,0,0]),',',int(scene_corners[1,0,1]))
    # print("C:",int(scene_corners[2,0,0]),',',int(scene_corners[2,0,1]))
    # print("D:",int(scene_corners[3,0,0]),',',int(scene_corners[3,0,1]))
    # ABCD = ([int(scene_corners[0,0,0]),int(scene_corners[0,0,1])],
    #         [int(scene_corners[1,0,0]),int(scene_corners[1,0,1])],
    #         [int(scene_corners[2,0,0]),int(scene_corners[2,0,1])],
    #         [int(scene_corners[3,0,0]),int(scene_corners[3,0,1])])

    ABCD = ([int(scene_corners[2,0,0]),int(scene_corners[2,0,1])],
            [int(scene_corners[3,0,0]),int(scene_corners[3,0,1])],
            [int(scene_corners[0,0,0]),int(scene_corners[0,0,1])],
            [int(scene_corners[1,0,0]),int(scene_corners[1,0,1])])    
    return(ABCD)

def Updown2dui1_PnP():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    ideal_Tvec=[[-15.0],[24.0],[192.0]]
    delta_Tvec=[[0],[0],[0]]

    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures

            # match(c)
            ABCD = match_once_printABCD1(c)    
            # print(ABCD)
            real_Tvec = PoseEstimate1(ABCD) 
            delta_Tvec = real_Tvec - ideal_Tvec
            if((delta_Tvec[1]>2) or (delta_Tvec[1]<-2)):
                if (delta_Tvec[1])>0:
                    # robot_move2_down()
                    robot_move2_down_N(1)
                    time.sleep(0.5)
                if (delta_Tvec[1])<0:
                    # robot_move2_up()
                    robot_move2_up_N(1)
                    time.sleep(0.5)
            else:
                print("Up and down is OK")
                break
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()

def LeftRight2dui1_PnP():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # ideal_Tvec=[[-19],[2],[276]]
    ideal_Tvec=[[-15.0],[24.0],[192.0]]
    delta_Tvec=[[0],[0],[0]]

    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures

            # match(c)
            ABCD = match_once_printABCD1(c)    
            # print(ABCD)
            real_Tvec = PoseEstimate1(ABCD) 
            delta_Tvec = real_Tvec - ideal_Tvec
            # print(delta_Tvec)
            if((delta_Tvec[0]>2) or (delta_Tvec[0]<-2)):
                if (delta_Tvec[0])>0:
                    robot_move0_right_N(1)
                    time.sleep(0.5)
                if (delta_Tvec[0])<0:
                    robot_move0_left_N(1)
                    time.sleep(0.5)
            else:
                print("Up and down is OK")
                break
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()


def ForwardBack2dui1_PnP():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    ideal_Tvec=[[-15.0],[24.0],[192.0]]
    delta_Tvec=[[0],[0],[0]]

    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures

            # match(c)
            ABCD = match_once_printABCD1(c)    
            # print(ABCD)
            real_Tvec = PoseEstimate1(ABCD) 
            delta_Tvec = real_Tvec - ideal_Tvec
            # print(delta_Tvec)
            if((delta_Tvec[2]>2) or (delta_Tvec[2]<-2)):
                if (delta_Tvec[2])>0:
                    robot_move1_forward_N(1)
                    time.sleep(0.5)
                if (delta_Tvec[2])<0:
                    robot_move1_back_N(1)
                    time.sleep(0.5)
            else:
                # move to the final ready position
                # move_to_open2_back20()
                print("Forward and Back is OK")
                break
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()

def Go_directly_PnP():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    ideal_Tvec=[[-16.56],[5.30],[284.50]]
    delta_Tvec=[[0],[0],[0]]

    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures

            # match(c)
            ABCD = match_once_printABCD1(c)    
            # print(ABCD)
            real_Tvec = PoseEstimate1(ABCD) 
            delta_Tvec = real_Tvec - ideal_Tvec
            # print(delta_Tvec)
            # print(delta_Tvec[2])
        
            n2 = delta_Tvec[2]/5.0 
            robot_move1_back_N(n2)

            n0 = delta_Tvec[0]/5.2
            robot_move0_right_N(n0)

            n1 = delta_Tvec[1]/4.5 
            robot_move2_down_N(n1)
            demo_start1 = [-0.1497819800013426, -0.13449279034770753, 0.42645890036500606, -0.015308508906232242, -2.323463252654852, -2.0578295936562165]
            move_to_pose_target(demo_start1)

            break
            
            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()