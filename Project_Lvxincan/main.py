#coding:utf-8
#!/usr/bin/env python3
# connect ethernet connection1 to control the robot 
import time
# import sys
# import math3d as m3d
# from aaa import *

from ForceDataGet import *
from Cv_detect import *
from RobotControl import *
from PoseEstimation12 import *
from svmMLiA_For_ForceData import *

# 经过测试，在光线良好的情况下，下面这段代码连续识别效果OK，可以取消注释后直接使用
# 可以作为图像采集的功能部分实现
def cap_display():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')

    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures

            match(c)
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

def cap_display_v1():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')

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
            Tvec = PoseEstimate1(ABCD) 
            print(Tvec)
            # print(T[0])
            # print(T[1])
            # print(T[2])
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

def open_insert():
    Go_directly_PnP()
    robot_move1_forward_N(38)
    robot_move0_right_N(5)
    robot_move1_back_N(12)
    robot_move0_left_N(5)
    robot_move1_forward_N(12)
    robot_move2_up_N(1.3)
    insert()



def cap_display5():
    cap = cv.VideoCapture(0)
    width = 1920
    height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c = 0
    fps = 100    # FPS of the video
    fourcc = cv.VideoWriter_fourcc(*'MJPG')

    while(cap.isOpened()):
        c = c + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv.imshow('Frame', frame)
            cv.imwrite('frame'+str(c) + '.jpg',frame)   # save as pictures

            # match(c)
            # Press Q on keyboard to  exit
            if c == 5:
                break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv.destroyAllWindows()

def insert():
    while(1):
        i=0
        for j in range(1,5):
            FM0=getForceData()
        input_insert=[((FM0[0]/10)**2+(FM0[1]/10)**2),((FM0[3]/2)**2+(FM0[4]/2)**2)]
        insert_predict=predict_res1(input_insert)
        if insert_predict==1:
            robot_move1_forward_N(2)
            i+=1
            if i>=4:
                break    #over

        else:
            if FM0[0]>=10:
                robot_move2_down(1)
            if FM0[0]<=-10:
                robot_move2_up_N(1)
            if FM0[1]>=10:
                robot_move0_left_N(0.5)
            if FM0[1]<=-10:
                robot_move0_right_N(0.5)



if __name__ == '__main__':
    # print(cv2.__version__)
    # print("hello")
    # cap_display_v1()        #capture the camera in real time
    # print(print_robot_joints())
    # robot_move2joints()
    # while True:
    #     print(getForceData())
    # robot_move1_forward_N(-38)
    # # confirmed to be OK for openning the cover on 4.15
    # robot_move2_up_N(4)
    # robot_move0_left_N(2)
    # robot_move1_back_N(2)

    open_insert()

    # Go_directly_PnP()
    # robot_move1_forward_N(38)
    # robot_move0_right_N(5)
    # robot_move1_back_N(12)
    # robot_move0_left_N(5)
    # robot_move1_forward_N(12)
    # robot_move2_up_N(1.3)
    # insert()



    # robot_move1_forward_N(8)
    # robot_move1_back_N(38)

    # robot_move2_up_N(-0.1)
    # robot_move0_left_N(0.1)

    # the next 12 funcitons confirmed to be OK
    # robot_move_joint1_add(1)
    # robot_move_joint1_sub(1)
    # robot_move_joint2_add(1)
    # robot_move_joint2_sub(1)
    # robot_move_joint3_add(1)
    # robot_move_joint3_sub(1)
    # robot_move_joint4_add(1)
    # robot_move_joint4_sub(-0.5)
    # robot_move_joint5_add(-0.2)
    # robot_move_joint5_sub(-1)
    # robot_move_joint6_add(1)
    # robot_move_joint6_sub(1)

    # robot_move0_left()
    # robot_move0_right()
    # robot_move1_forward()
    # robot_move1_back()
    # robot_move2_up()
    # robot_move2_down()
    # robot_move3_roll_add()
    # robot_move3_roll_sub()
    # robot_move4_pitch_add()
    # robot_move4_pitch_sub()
    # robot_move5_yaw_add()
    # robot_move5_yaw_sub()

    # robot_move1_forward_N(-2)
    # robot_move1_back_N(10)
    # robot_move2_up_N(1)
    # robot_move2_down_N(1)
    # robot_move0_left_N(2)
    # robot_move0_right_N(5)

    # robot_move1_back_N(12)
    # back initial
    # move_to_tie_back40()
    # robot_move2_up_N(2)
    # robot_move0_left_N(2)
    
    # Updown2dui1_PnP()
    # LeftRight2dui1_PnP()
    # ForwardBack2dui1_PnP()

    # robot_move1_forward_N(0.2)

    # teach 4.16
    # pose_tie_back20 = [-0.14950710343362192, -0.043985117548752395, 0.4264057253525786, -0.01568110587832934, -2.3237839956314117, -2.058070451668305]
    # move_to_pose_target(pose_tie_back20)
    # robot_move1_forward_N(10)

    #teach1
    # teach1 =  [-0.13856815028504782, 0.08117438881800951, 0.42709057531762856, 0.029835059768004386, -2.2941801572458247, -2.0627403435792675]
    # move_to_pose_target(teach1)
    #teach2
    # robot_move0_left_N(-0.1)
    # robot_move1_forward_N(0.1)
    # robot_move2_up_N(-0.1)
    # teach2 =  [-0.14099635917010614, 0.08322942695151692, 0.42318835658608994, 0.046159323895640335, -2.2839260305273297, -2.0785959185752323]
    # move_to_pose_target(teach2)
    # robot_move_joint4_add(-0.1)
    # robot_move_joint3_add(-0.1)
    # robot_move2_up_N(-0.1)
    #teach3
    # print_robot_tcp()

    # print(getForceData())
    # robot_move1_forward_N(0.5)
    # move_to_tie_back40()

    # Updown2dui1_PnP() # confirmed to be OK
    # print("OK")
    # LeftRight2dui1_PnP()
    # print("OK")
    # ForwardBack2dui1_PnP()
    # print("OK")

#     # to debug in 2.26
    # print_robot_tcp()

# # return to the initial pos set before 
#     robot_move1_back20times()
#     # for i in range(1,21):
#     #     robot_move1_back()

#     print("The operation is OK!! ")        

# Pose1
# Pose2 = pose1+[0,0,20times0.001,0,0,0]
# pose move right to open the gate
# pose move to be ready for the insert
# pose move to leave the socket
# pose move to be ready to close the gate
# pose move to leave the socket with the plug


    # ABCD = ([878 , 425], [1141 , 429], [1149 , 619], [873 , 626])
    # PoseEstimate(ABCD)

    # for i in range(1,21):
    #     print(i)
    #     robot_move1_back()
    #     robot_move1_forward()

    #rem4 start
    # robot_move1_forward_N(10)
    # robot_move1_forward_less()
    # robot_move0_right_N1(6)
    # robot_move1_back_N(6)
    # robot_move0_left_N(5)
    # robot_move1_forward_N(3)

    # robot_move1_back_N(10)
    # robot_move2_up_N(4)
    # robot_move2_down_N(1)
    # robot_move0_left_N(3)
    # robot_move0_right_N(1)

    #right6,back6,left5,forward6,up4
    # duikong1 robot_move1_forward_N(4)
    # robot_move2_up_less()#2
    # robot_move0_left_less()

    #rem2 start
    # robot_move1_forward_N(10)
    # robot_move0_right_N1(6)
    # robot_move1_forward()#3
    # robot_move1_back_N(6)
    # robot_move0_left_N(3)
    # robot_move1_forward_N(5)

    # robot_move_forward_less()
    # robot_move1_forward() # forward 0.01
    # robot_move1_back()  # 9times

    # robot_move0_left()    #left 0.005
    # robot_move0_right() 

    # robot_move2_up()    # up 0.005
    # robot_move2_down()

    # get_Transformation()
    # print_robot_tcp()

    # move_to_open1_back20()  # to debug

    #to use PnP algorithm and get the result

    # robot_move3_roll_add()
    # robot_move3_roll_sub()
    # robot_move4_pitch_add()
    # robot_move4_pitch_sub()
    # robot_move5_yaw_add()
    # robot_move5_yaw_sub()

    # cap_display5()
    # ABCD = match_once_printABCD()
    # print(ABCD)
    # Tvec = PoseEstimate1(ABCD)

    # Rx,Ry,Rz have been confirmed to be OK

    # robot_move1_forward()
    # robot_move1_back()   
    # robot_move3_roll_add()
    # robot_move3_roll_sub()
    # robot_move4_pitch_add()
    # robot_move4_pitch_sub()
    # robot_move5_yaw_add()
    # robot_move5_yaw_sub()

    # get_Transformation()
    # print_robot_tcp()
    # 基本pose1,2,3位置运动测试ok
    # robot_move0_left()
    # robot_move0_right()
    # robot_move1_forward()
    # robot_move1_back()
    # robot_move2_up()
    # robot_move2_down()

    # match1()
    # move_to_dui1()
    # robot_move_task()
    # print_robot_tcp()
    # robot_move_up()
    # robot_move_forward1()   
    # rotate_forward()
    # a = 0
    # while(1):
    #     print(a)
    #     get_Transformation()
    #     time.sleep(1)
    #     a += 1

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

#2019.2.22 open1 pose remember
# robot tcp is at:  [-0.1762626917312132, 0.025876025826768925, 0.4184888775492209, 0.07580816880297646, -2.282136007463382, -2.111546737616171]

#2019.2.22 open1 move back 20 times pose remember
# robot tcp is at:  [-0.1751431725350633, -0.1660739108733619, 0.4115460201064948, 0.0507198593929355, -2.313324220394815, -2.0747100828322385]
