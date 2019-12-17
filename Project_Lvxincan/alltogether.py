#coding=utf-8 
from __future__ import print_function
import cv2 as cv
import numpy as np
from matplotlib import pylab as plt
import argparse
import glob
from math import pi
import urx
import logging
import time
import sys
import math3d as m3d   
def move_to_dui1():
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))
    v = 0.02
    a = 0.1      
    #dui1 pose start
    pose = [-0.015371047712964686, 0.16291898650384315, 0.3583823679602556, 0.729218924720313, 1.756045637777791, 1.72787322187088]
    rob.movel(pose, acc=a, vel=v)
    rob.close()

def robot_move_OnePose():
    # to move up
    print("moving to a known pose")
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))

    v = 0.01
    a = 0.1

    pose = rob.getl()
    print("robot tcp is at: ", pose)    #   in the world coordinate position and rotation

    # move in z direction up for 0.02
    # print("moving in z")

    pose[0] = 0.1
    pose[1] = 0.1
    pose[2] = 1.0
    rob.movel(pose, acc=a, vel=v)
    rob.close()

def rotate_forward():
    print("rotate_forward")
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))

    v = 0.02
    a = 0.1

    pose = rob.getl()
    print("robot tcp is at: ", pose)    #   in the world coordinate position and rotation

    # move in x direction  for 0.02
    # print("moving in x")
    pose[4] += pi/90
    rob.movel(pose, acc=a, vel=v)
    rob.close()

def rotate_left():
    print("rotate_left")

def rotate_up():
    print("rotate_up")


def robot_move_right1(baselength):
    # to move right l unit:mm
    print("moving right")
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))

    v = 0.02
    a = 0.1

    pose = rob.getl()
    print("robot tcp is at: ", pose)    #   in the world coordinate position and rotation

    # move in x direction  for 0.02
    # print("moving in x")
    pose[0] += baselength
    rob.movel(pose, acc=a, vel=v)
    rob.close()
def robot_move_forward1():
    # to move forward
    print("moving forward")
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))

    v = 0.02
    a = 0.1

    pose = rob.getl()
    print("robot tcp is at: ", pose)    #   in the world coordinate position and rotation

    # move in y direction  for 0.02
    # print("moving in y")
    pose[1] -= 0.02
    rob.movel(pose, acc=a, vel=v)
    rob.close()



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
    center_point0=(int(scene_corners[0,0,0] + img_object.shape[1])+int(scene_corners[2,0,0] + img_object.shape[1]))/2
    center_point1=(int(scene_corners[0,0,1])+int(scene_corners[2,0,1]))/2
    center_point=[center_point0,center_point1]
    print(center_point)
    return(center_point)

def match1():   
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='rec_small1.jpg')
    parser.add_argument('--input2', help='Path to input image 2.', default='facing1.jpg')
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
    # cv.imwrite('Good Matches&Object detection1.jpg',img_matches)
    # cv.imshow('Good Matches&Object detection1.jpg',img_matches)
    # print(int(scene_corners[0,0,0]),img_object.shape[1],int(scene_corners[0,0,1]))
    # print('Good Matches&Object detection1'+str(c) + '.jpg')
    center_point0=(int(scene_corners[0,0,0] + img_object.shape[1])+int(scene_corners[2,0,0] + img_object.shape[1]))/2
    center_point1=(int(scene_corners[0,0,1])+int(scene_corners[2,0,1]))/2
    center_point=[center_point0,center_point1]
    # return(center_point)
    print(center_point)
    while(True):
    # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break        

def move_point():
    # to move up
    print("moving to the point")
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))
    v = 0.02
    a = 0.1
    #close1-6
    # pose =  [0.11331817392983798, 0.2801514404360457, 0.40716238831066137, 0.7288518769893056, 1.75720949766542, 1.7288762714958366]
    # rob.movel(pose, acc=a, vel=v)
    rob.close()    

def robot_move_task():
    # robot_move_task_from_ReferencePosition
    # to move as the task asked
    print("moving to complate the task")
    rob = urx.Robot("192.168.80.2")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))

    l = 0.02
    v = 0.02
    a = 0.1

    pose = rob.getl()
    print("robot tcp is at: ", pose)    #   in the world coordinate position and rotation

    #dui1 pose start
    # pose = [-0.015371047712964686, 0.16291898650384315, 0.3583823679602556, 0.729218924720313, 1.756045637777791, 1.72787322187088]
    # rob.movel(pose, acc=a, vel=v)
    # step1
    pose = [0.09843593470393579, 0.26510182111053154, 0.4028641488364185, 0.7293840999220157, 1.7575432907350066, 1.7280083427019985]
    rob.movel(pose, acc=a, vel=v)
    # step2
    pose =  [0.11349549374315637, 0.2802989236507926, 0.40720806884446564, 0.7290357525244547, 1.7569778326154977, 1.7287120933853668]
    rob.movel(pose, acc=a, vel=v)
    # step3
    pose =  [0.11460174179046292, 0.28153593539694216, 0.4074298765968609, 0.7295337903752471, 1.756310919064192, 1.7283559296692148]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.12923785534015672, 0.2671670032288611, 0.40747408164237053, 0.7295000886040007, 1.7572500353651812, 1.7283636977205024]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.11078068722321228, 0.25527572034962204, 0.40764048056389807, 0.7298588022893313, 1.7555891125498237, 1.7284234936835599]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.07648318951482813, 0.22131664005025137, 0.4077336729334317, 0.7290399452789309, 1.7566098675466841, 1.7290427412939582]
    rob.movel(pose, acc=a, vel=v)
    time.sleep(2) 
    #in 1-4
    pose =  [0.1096277417105041, 0.2782563939867772, 0.40945510584463063, 0.7304462462673038, 1.7569070677822813, 1.7268994057911478]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.11813817891258888, 0.2856367025277646, 0.41328282986770337, 0.7293913095346095, 1.757686739360187, 1.7281365277259495]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.11939772214092868, 0.2885459066988721, 0.411258361377801, 0.729510577813205, 1.7552934806593876, 1.7289513919241533]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.12358349828409654, 0.29316420912719965, 0.410826266327851, 0.7290838685610902, 1.757399470553258, 1.7285771052298267]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.13147317314094595, 0.3011585845734863, 0.41078134124622057, 0.728876103815376, 1.7572176942540814, 1.7289702636115485]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.1405024103151181, 0.310255323239094, 0.4110833042894985, 0.7287406831381755, 1.7573406460681256, 1.7291010739354038]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.14239719241844506, 0.31252705462501135, 0.41209069321252606, 0.7291499575488868, 1.7628837802632797, 1.7265063352354237]
    rob.movel(pose, acc=a, vel=v)
    time.sleep(5)
    #out 2-6
    pose =  [0.1334431445545815, 0.3026290175057595, 0.4118846240821475, 0.730741243829314, 1.760253414711346, 1.7255141352815433]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.12437409880672887, 0.29382238237474473, 0.41159802946857316, 0.7305493146697993, 1.7623292137236457, 1.724746836319876]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.11798124860681335, 0.28735259472234725, 0.4115068046183376, 0.7308702421744679, 1.7625799214084237, 1.7244085447547393]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.08526943020955498, 0.2547964393826721, 0.4116081134616931, 0.7302656953431732, 1.761960373435015, 1.7250979247124252]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.0592818549986954, 0.22880215454237002, 0.4115284040360025, 0.7307364002519257, 1.7630925285669465, 1.7241497309932778]
    rob.movel(pose, acc=a, vel=v)
    #close1-6
    pose =  [0.03907648121863697, 0.24824147478697123, 0.4113091730563136, 0.7301457328402742, 1.7633817788209647, 1.724228231133081]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.020770591111322328, 0.2668176069070978, 0.4114559310892153, 0.730043416937704, 1.7621468652746528, 1.7246688495323277]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.007834567566814908, 0.2799198500316586, 0.41113026596905333, 0.7296730780348917, 1.763338816584342, 1.7238073259520172]
    rob.movel(pose, acc=a, vel=v)
    pose =  [-0.002584685455172076, 0.29037234929837574, 0.41119475375580194, 0.7297502879408158, 1.7632873208189492, 1.7238532499992436]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.04506127505101218, 0.33840389742311, 0.41116776085679263, 0.7291835964187542, 1.7638554918217029, 1.724986842882753]
    rob.movel(pose, acc=a, vel=v)
    pose =  [0.09526823207181682, 0.263550292508512, 0.41155533470618194, 0.729797114405243, 1.7631994432180738, 1.7253466398567225]
    rob.movel(pose, acc=a, vel=v)
    time.sleep(2)
    pose =  [0.11489694220364684, 0.2821064908742386, 0.406623292470202, 0.7287839192148127, 1.7599187617629883, 1.7278966961919806]
    rob.movel(pose, acc=a, vel=v)
    #close OK
    time.sleep(2)
    #return to the original point
    pose =  [-0.026091850507074233, 0.13637172277017537, 0.34192834637749725, 0.7168010678810264, 1.9995058927833398, 1.4523042091915035]
    rob.movel(pose, acc=a, vel=v)

    # rob.movel(pose, acc=a, vel=v)
    rob.close()    

def Updown2dui1():
    cap = cv.VideoCapture(0)
    c = 0
    fps = 10   # FPS of the video
    # rval=cap.isOpened()
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # the last parameter is the size
    # videoWriter = cv.VideoWriter('saveVideo.avi', fourcc, fps, (640, 480))
    # Read until video is completed
    # ideal_center_point=[600,242]
    ideal_center_point=[450,211]
    delta_point=[0,0]
    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures
            # frame1 = frame

            # print("find the dui1 pose")
            real_center_point=match(c)
            delta_point[0]=real_center_point[0]-ideal_center_point[0]
            delta_point[1]=real_center_point[1]-ideal_center_point[1]
            print(delta_point[0])

            if((delta_point[1]>15)|(delta_point[1]<-15)):
                if (delta_point[1])>0:
                    robot_move_down()
                if (delta_point[1])<0:
                    robot_move_up()
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

def LeftRight2dui1():
    cap = cv.VideoCapture(0)
    c = 0
    fps = 10   # FPS of the video
    # rval=cap.isOpened()
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # the last parameter is the size
    # videoWriter = cv.VideoWriter('saveVideo.avi', fourcc, fps, (640, 480))
    # Read until video is completed
    # ideal_center_point=[600,242]
    ideal_center_point=[450,211]
    delta_point=[0,0]
    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures
            # frame1 = frame

            # print("find the dui1 pose")
            real_center_point=match(c)
            delta_point[0]=real_center_point[0]-ideal_center_point[0]
            delta_point[1]=real_center_point[1]-ideal_center_point[1]
            print(delta_point[0])

            if(((delta_point[0]>15))|((delta_point[0])<-15)):
                if delta_point[0]>0:
                    robot_move_right()
                if delta_point[0]<0:
                    robot_move_left()  
            else:
                print("Left and right is OK")
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

def ForwardBack2dui1():
    cap = cv.VideoCapture(0)
    c = 0
    fps = 10   # FPS of the video
    # rval=cap.isOpened()
    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # the last parameter is the size
    # videoWriter = cv.VideoWriter('saveVideo.avi', fourcc, fps, (640, 480))
    # Read until video is completed
    ideal_center_point=[600,242]
    delta_point=[0,0]
    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures
            # frame1 = frame

            # print("find the dui1 pose")
            real_center_point=match(c)
            delta_point[0]=real_center_point[0]-ideal_center_point[0]
            delta_point[1]=real_center_point[1]-ideal_center_point[1]
            print(delta_point[0])

            if((delta_point[1]>15)|(delta_point[1]<-15)):
                if (delta_point[1])>0:
                    robot_move_down()
                if (delta_point[1])<0:
                    robot_move_up()
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





# pose_dui1 =  [-0.015379972627099047, 0.16278659829816503, 0.35778023133243914, 0.7293855033267473, 1.7571449945285949, 1.7263708455148004]



# if __name__ == '__main__':
    # match1()
    # move_to_dui1()
    # robot_move_task()
    # print_robot_tcp()
    # robot_move_up()
    # robot_move_forward1()   
    # rotate_forward()
    # get_Transformation()

    # 此处输出的Orientation为RxRyRz旋转矩阵相乘的出来的结果
    # pose = [-0.015379972627099047, 0.16278659829816503, 0.35778023133243914, 0.7293855033267473, 1.7571449945285949, 1.7263708455148004]
    # pose = [1.57, 0, 0, 1.57, 0, 0]
    # print(m3d.Transform())
    
    # robot_move_right1(0.01)   # 最多只能移动50cm
    # delta_robot_tcp1 = print_delta_robot_tcp() 
    # delta_robot_tcp1 =  [-8.38816606e-05, 4.68648427e-05, 4.68079196e-04, -2.88025478e-02, -7.31398710e-02, 5.52882549e-02]   
    # test_delta_robot_tcp(delta_robot_tcp1)  # test_delta_robot_tcp() function OK



# Teach and peg in the hole_OK
    # Updown2dui1()
    # time.sleep(3)
    # LeftRight2dui1()
    # time.sleep(3)
    # robot_move_task()
