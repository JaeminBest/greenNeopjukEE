# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from detection.setting_opencv import setting, construct_cord, selection, resetting
from detection.calibration import calibration, transform
from detection.measure import position, speed
import cv2
import numpy as np
import os
from os.path import isfile, join
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

# initial dir setting for debugging
rootdir = './detection/data/'
datadir = '1.jpg'

# debug function
def open(rootdir=rootdir,datadir=datadir):
    # file validation    
    dir = os.path.join(rootdir,datadir)
    if (os.path.isfile(dir)):
        img = cv2.imread(dir,cv2.IMREAD_COLOR)
        return img
    else:
        return None


# record n frames and do calibration to set calibration value(deg,scale,cross,central,side)
# RETURN : new_param for detection
# reference : keras-yolo3
def calibRecord(video_path, output_path = "", n=5):
    # list of setting json obj
    res_lst = []

    # video capture
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    
    print("=========================================")
    print("=========== doing calibration ===========")
    print("=========================================")
    start_time = prev_time
    end_time = prev_time
    toal_exec_time = end_time-start_time
    cnt = 0
    while (total_exec_time<n):
        cnt+=1
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        # calbration start
        res = setting(image)
        if (not res):
            res_lst.append(res)
        curr_time = timer()
        end_time = curr_time
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        toal_exec_time = end_time-start_time

    if len(res_lst)<cnt/2:
        # we need to re-set threshold of all other things
        print("=========================================")
        print("========== calibration FAILURE ==========")
        print("=========================================")
        return None

    # select the most frequent parameter
    res = selection(image,res_lst)
    cross = res[0]
    center = res[1]
    side = res[2]

    # re-setting all of the parameter
    temp_res = resetting(image,cross,center,side)
    fn_res = calibration(image,temp_res)
    print("=========================================")
    print("=========== calibration done ============")
    print("=========================================")

    return fn_res


# pipeline for yolo NN input
# at yolo.py line 192-193
def pipe_yolo(image, param):
    rimg = transform(image,param)
    #cv2.imshow('main_transformed',cv2.resize(rimg,dsize=(1200,600)))
    return rimg

# pipeline for SUMO net
# input param, objs(result from yolo NN)
def pipe_sumo(param, objs):
    return

# measure all data from video and then send it to server 
def main(mode = 0, video_path="", output_path = ""):
    param = None
    cord3 = None
    cord2 = None
    while True:
        if (mode==0):   # stay mode
            continue
        
        if (mode==1):   # calibration mode
            if (param is None):
                param = calibRecord(video_path)
                cords = construct_cord(param)
                cord3 = cords[0]
                cord2 = cords[1]    
                print(param)
        
        if (mode==2):   # detecting mode
            if (param is None):
                print("calibration start")
                mode = 1
                continue

            # for testing...
            # open => need to be changed to cv2.videocapture
            img = open(rootdir,datadir)
            cv2.imshow('main_org',cv2.resize(img,dsize=(1200,600)))
            cv2.waitKey()

            #### yolo-keras source code ######
            # yolo-net detection (collabo with YOLO)
            # INPUT : rimg (rotated, scaled image)
            # OUTPUT : json object of object class/pixel value of boundary
            ##################################
            #objs = blahblahblah
            ##################################

            
            # objs form of 
            objs = None
            objs = pipe_yolo(img,param)
            
            # draw point to rimg (collabo with SUMO)
            # INPUT: json object
            # OUTPUT : position value basis is line of crosswalk
            # sub-OUTPUT : coordinate image (2D,3D)
            ##########################
            # pipe_sumo()
            send_msg = []
            for obj in objs:
                temp = position(obj,param,cord3,cord2)
                send_msg.append(temp)
            print(send_msg)
            ##########################
    return True

if __name__=='__main__':
    main()