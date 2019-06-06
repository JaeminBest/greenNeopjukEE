# @Author  : JaeminBest
# @File    : detection/measure.py
# @IDE: Microsoft Visual Studio Code
import cv2
import numpy as np
import math
from detection.setting_opencv import get_intersect
from detection.calibration import scaP,rotP

# param : initial calibration param(central line, side line, crosswalk...)
# key description below:

def transP(point,param):
    # point is form of [[x,y]]
    origin = [[param['shape'][0]/2,param['shape'][1]/2]]
    scale = param['scale']
    degree = param['deg']
    newP = scaP(point,scale,origin)
    newP = rotP(point,degree,origin)
    return newP

# INPUT : obj(element in object list, form of (label,(left,top),(right,bottom))), 
#         param(calibration returned parameter), cord3(3D coordination that central, 
#         side, crosswalk line drawn), cord2(2D version of cord3)
# OUTPUT : position dictionary object having key as 'lane', 'distance'
#          in here, distance is normal distance between crosswalk and object
# function : calculate position of obj given
def position(objs, param, cord3, cord2):
    # obj form : (label, (left, top), (right, bottom))
    reses = []
    n_person = 0
    for obj in objs:
        print('obj0',obj[0])
        if 'person' in obj[0]:
            n_person+=1
        else:
            left = obj[1][0]
            top = obj[1][1]
            right = obj[2][0]
            bottom = obj[2][1]

            # we will asume position of car be bottome line of detected box
            x = (left+right)/2.
            y = bottom

            pt = [[x,y]]
            pt = np.array(pt,dtype=np.float32)
            pt = np.array([pt])

            cv2.circle(cord3, (int(x),int(y)), 20, (0,255,0),-1)
            new_pt = cv2.perspectiveTransform(pt, param['persM'])
            
            nx = new_pt[0][0][0]
            ny = new_pt[0][0][1]
            cv2.circle(cord2, (int(nx),int(ny)), 20, (0,255,0),-1)        
            res = dict()
            # lane determination
            if (math.fabs(nx-10)>math.fabs(nx-50)):
                print("2nd lane")
                res['lane'] = 2
            else:
                print("1st lane")
                res['lane'] = 1

            # distance determination
            res['distance'] = (nx-param['trn_cross'])*param['grid']
            reses.append(res)

    cv2.imshow('test_obj_3D', cv2.resize(cord3,dsize=(1200,600)))
    cv2.imshow('test_obj_2D', cord2)

    return reses, n_person

# INPUT : obj(element in object list, form of (label,(left,top),(right,bottom))), 
#         param(calibration returned parameter), cord3(3D coordination that central, 
#         side, crosswalk line drawn), cord2(2D version of cord3)
# OUTPUT : position dictionary object having key as 'lane', 'distance'
#          in here, distance is normal distance between crosswalk and object
# function : calculate position of obj given
def speed(objs, res_lst):
    # obj is form of [[x1,y1,x2,y2]]
    # res_lst is n frame list of position json object 
    sm = 0
    cnt = 0
    for res in res_lst:
        sm += res['distance']
        cnt+=1
    speed = sm/cnt
    return -speed
