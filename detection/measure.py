# @Author  : JaeminBest
# @File    : detection/measure.py
# @IDE: Microsoft Visual Studio Code
import cv2
import numpy as np
import math
from detection.setting_opencv import get_intersect, slope
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

def belowLine(point,line):
    # point form of [[x,y]]
    # line form of [[x1,y1,x2,y2]]
    slop = slope(line)
    for x,y in point:
        a = x
        b =y

    for x1,y1,x2,y2 in line:
        x1 = x1
        y1 = y1
        x2 = x2
        y2 = y2

    yp = slop*(a-x1)+y1
    if (yp<b):
        return True
    else:
        return False

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
    threshold = 0
    clen = param['trn_bump']/2.
    flag = False
    for obj in objs:
        print('obj :',obj[0])
        if 'person' in obj[0]:
            n_person+=1
            left = obj[1][0]
            top = obj[1][1]
            right = obj[2][0]
            bottom = obj[2][1]

            # we will asume position of car be bottome line of detected box
            x = (left+right)/2.
            y = bottom

            if (belowLine([[left,y]],param['center'])):
                #print("below line")
                continue

            pt1 = [[left,bottom]]
            pt2 = [[right,bottom]]
            pt1 = np.array(pt1,dtype=np.float32)
            pt1 = np.array([pt1])
            pt2 = np.array(pt2,dtype=np.float32)
            pt2 = np.array([pt2])

            cv2.circle(cord3, (int(x),int(y)), 20, (0,255,0),-1)
            new_pt1 = cv2.perspectiveTransform(pt1, param['persM'])
            new_pt2 = cv2.perspectiveTransform(pt2, param['persM'])
            
            hi = param['afterRegion'][2][1]
            ro = param['afterRegion'][0][1]

            nx = (new_pt1[0][0][0]+new_pt2[0][0][0])/2
            ny = (new_pt1[0][0][1]+new_pt2[0][0][1])/2
            res = dict()
            
            if (ny>=hi+threshold):
                continue
            if (ny<=ro-threshold):
                continue
            
            # lane determination
            if ((nx<=param['trn_cross']+clen) and (nx >= param['trn_cross']-clen)):
                #print("2nd lane")
                flag = True

        else:
            left = obj[1][0]
            top = obj[1][1]
            right = obj[2][0]
            bottom = obj[2][1]

            # we will asume position of car be bottome line of detected box
            x = (left+right)/2.
            y = bottom

            if (belowLine([[left,y]],param['center'])):
                print("below line")
                continue

            pt1 = [[left,bottom]]
            pt2 = [[right,bottom]]
            pt1 = np.array(pt1,dtype=np.float32)
            pt1 = np.array([pt1])
            pt2 = np.array(pt2,dtype=np.float32)
            pt2 = np.array([pt2])

            cv2.circle(cord3, (int(x),int(y)), 20, (0,255,0),-1)
            new_pt1 = cv2.perspectiveTransform(pt1, param['persM'])
            new_pt2 = cv2.perspectiveTransform(pt2, param['persM'])
            
            hi = param['afterRegion'][2][1]
            ro = param['afterRegion'][0][1]

            nx = (new_pt1[0][0][0]+new_pt2[0][0][0])/2
            ny = (new_pt1[0][0][1]+new_pt2[0][0][1])/2
            res = dict()
            # lane determination
            if (math.fabs(ny-ro)>math.fabs(ny-hi)):
                print("2nd lane")
                res['lane'] = 2
                conv_ny = 40
            else:
                print("1st lane")
                res['lane'] = 1
                conv_ny = 20
            
            if (ny>=hi+threshold):
                continue
            if (ny<=ro-threshold):
                continue

            cv2.circle(cord2, (int(nx),int(conv_ny)), 20, (0,255,0),-1)     

            # distance determination
            res['distance'] = (nx-param['trn_cross'])*param['grid']
            reses.append(res)

    cv2.imshow('test_obj_3D', cv2.resize(cord3,dsize=(1200,600)))
    cv2.imshow('test_obj_2D', cord2)

    return reses, n_person, flag

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
