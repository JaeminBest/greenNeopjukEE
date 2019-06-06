# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from detection.setting_opencv_east import construct_cord, selection, setting, find
from detection.calibration import calibration, transform
from detection.measure import position, speed
import cv2
import numpy as np
import os
from os.path import isfile, join
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

# debug function
def imopen(dir=None):
    # file validation    
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
        found = find(image)
        if (not found):
            res_lst.append(found)
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
    found = selection(image,res_lst)

    # re-setting all of the parameter
    temp_res = setting(found)
    fn_res = calibration(image,res=temp_res)
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