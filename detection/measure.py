# @Author  : JaeminBest
# @File    : detection/measure.py
# @IDE: Microsoft Visual Studio Code
import cv2
import numpy as np
import math
from detection.setting_opencv import get_intersect

# param : initial calibration param(central line, side line, crosswalk...)
# objs : list of objs and pixel value

# calculate position of obj given
def position(obj, param, cord3, cord2):
    x1 = obj[0][0]
    y1 = obj[0][1]
    x2 = obj[1][0]
    y2 = obj[1][1]

    x = (x1+x2)/2.
    y = (y1+y2)/2.

    pt = [[x,y]]
    pt = np.array(pt,dtype=np.float32)
    pt = np.array([pt])

    cv2.circle(cord3, (int(x),int(y)), 20, (0,255,0),-1)
    cv2.imshow('test_obj_3D', cv2.resize(cord3,dsize=(1200,600)))
    new_pt = cv2.perspectiveTransform(pt, param['persM'])
    
    nx = new_pt[0][0][0]
    ny = new_pt[0][0][1]
    cv2.circle(cord2, (int(nx),int(ny)), 20, (0,255,0),-1)
    cv2.imshow('test_obj_2D', cord2)
    
    res = dict()
    # lane determination
    if (math.fabs(nx-10)>math.fabs(nx-50)):
        print("2nd lane")
        res['lane'] = 2
        return 
    else:
        print("1st lane")
        res['lane'] = 1

    # distance determination
    res['distance'] = (nx-param['trn_cross'])*param['grid']
    return res


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