from detection import calibRecord, pipe_yolo, pipe_sumo, open
from detection.setting_opencv import construct_cord
from detection.measure import position
from yolo import yolo, yolo_video
import cv2

# initial dir setting for debugging
rootdir = './detection/data/'
datadir = '1.jpg'

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