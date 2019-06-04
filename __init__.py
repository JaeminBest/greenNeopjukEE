# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from detection.setting_opencv import setting, construct_cord, selection, resetting
from detection.calibration import calibration, transform
from detection.measure import position, speed
#from yolo import detect_video
import cv2
import numpy as np
import os
from os.path import isfile, join
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

import sys
import argparse
from yolo.yolo import YOLO, detect_video
from PIL import Image
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
    total_exec_time = end_time-start_time
    cnt = 0
    while (total_exec_time<n):
        cnt+=1
        return_value, frame = vid.read()
        if return_value == False:
            print("false-----------------")
        else:
            print("true_-----------------") 
    #    pil_image = Image.fromarray(frame)
        # calbration start
        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame)
     #   pil_image.show()
        print("-----print image-----")
        print(pil_image)
        print(type(pil_image))
        # convert to opencv image
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
   # 2    pil_image = Image.open(pil_image).convert('RGB') 
   # 2    image = np.array(pil_image) 
        # Convert RGB to BGR 
   # 2    image = image[:, :, ::-1].copy()
        cv2.imshow('test', cv2.resize(image,dsize=(1200,600)))
        print("-----print convert image------")
        print(image)
        print(type(image))
        res = setting(image)
#        res = setting(frame)
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
        total_exec_time = end_time-start_time

    if len(res_lst)<cnt/2:
        # we need to re-set threshold of all other things
        print("=========================================")
        print("========== calibration FAILURE ==========")
        print("=========================================")
        return None

    # select the most frequent parameter
    res = selection(image,res_lst)
#    res = selection(frame,res_lst)
    cross = res[0]
    center = res[1]
    side = res[2]

    # re-setting all of the parameter
    temp_res = resetting(image,cross,center,side)
    fn_res = calibration(image,temp_res)
 #   temp_res = resetting(frame,cross,center,side)
 #   fn_res = calibration(frame,temp_res)
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
def main(yolo, video_path="", output_path = "", mode = 0):
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
            #trn_img = pipe_yolo(img,param)
            detect_video(yolo, param, video_path, output_path)

            objs = None
            #### yolo-keras source code ######
            # yolo-net detection (collabo with YOLO)
            # INPUT : rimg (rotated, scaled image)
            # OUTPUT : json object of object class/pixel value of boundary
            ##################################
            #objs = blahblahblah
            ##################################

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

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__=='__main__':
        # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        main(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output, mode = 1)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
