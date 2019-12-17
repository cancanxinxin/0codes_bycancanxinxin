# object detection real time    confirmed to be OK
#coding=utf-8 
import cv2 as cv
import numpy as np
from matplotlib import pylab as plt
import argparse
import glob
#  it's good enough to use

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
        print(int(scene_corners[0,0,0]),img_object.shape[1],int(scene_corners[0,0,1]))
        # print('Good Matches&Object detection1'+str(c) + '.jpg')

cap = cv.VideoCapture(0)
c = 0
fps = 100    # FPS of the video
# rval=cap.isOpened()
fourcc = cv.VideoWriter_fourcc(*'MJPG')
# the last parameter is the size
# videoWriter = cv.VideoWriter('saveVideo.avi', fourcc, fps, (640, 480))
# Read until video is completed

while(cap.isOpened()):
    c = c + 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        cv.imshow('Frame', frame)
        cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures
        # frame1 = frame

        match(c)
        # cv.imwrite('Good Matches&Object detection1'+str(c) + '.jpg', frame1)  # save as pictures
        # cv.imshow('Frame1', frame1)
        # videoWriter.write(frame1)
        # print(frame)

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


