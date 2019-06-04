# @Author  : JaeminBest
# @File    : detection/calibration.py
# @IDE: Microsoft Visual Studio Code

import cv2
import numpy as np
import math
from detection.setting_opencv import calc_setting, setting, find

# transform point by rotating about degree
def rotP(point,degree, origin):
    # point is form of [[x1,y1]]
    # degree is form of tan(angle), same as slope
    angle = math.atan2(degree,1.0)
    
    for a,b in point:
        x = a 
        y = b

    for a,b in origin:
        offset_x = a
        offset_y = b

    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return [[int(qx), int(qy)]]

# transform line by rotating about degree
def rotL(line,degree, origin):
    # line is form of [[x1,y1,x2,y2]]
    # degree is form of tan(angle), same as slope
    for x1,y1,x2,y2 in line:
        p1 = [[x1,y1]]
        p2 = [[x2,y2]]

    new_p1 = rotP(p1,degree,origin)
    new_p2 = rotP(p2,degree,origin)
    
    for a,b in new_p1:
        nx1 = a
        ny1 = b
    
    for a,b in new_p2:
        nx2 = a
        ny2 = b

    new_line = [[nx1,ny1,nx2,ny2]]

    return new_line

# transform point by rotating about degree
def scaP(point,scale, origin):
    # point is form of [[x1,y1]]
    
    for a,b in point:
        x = a 
        y = b

    for a,b in origin:
        offset_x = a
        offset_y = b

    adjusted_x = (x - offset_x)*scale
    adjusted_y = (y - offset_y)*scale
    cos_rad = math.cos(0)
    sin_rad = math.sin(0)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return [[int(qx), int(qy)]]

# transform line by rotating about degree
def scaL(line,scale, origin):
    # line is form of [[x1,y1,x2,y2]]
    # degree is form of tan(angle), same as slope
    for x1,y1,x2,y2 in line:
        p1 = [[x1,y1]]
        p2 = [[x2,y2]]

    new_p1 = scaP(p1,scale,origin)
    new_p2 = scaP(p2,scale,origin)
    
    for a,b in new_p1:
        nx1 = a
        ny1 = b
    
    for a,b in new_p2:
        nx2 = a
        ny2 = b

    new_line = [[nx1,ny1,nx2,ny2]]

    return new_line

def rotation(img,param):
    cols = img.shape[0]
    rows = img.shape[1]
    dst = cv2.warpAffine(img,param['rotM'],(rows,cols))
    return dst

def scaling(img,param):
    h = img.shape[0]
    w = img.shape[1]
    trn = param['scale']-1
    res = cv2.resize(img,None,fx=param['scale'], fy=param['scale'], interpolation = cv2.INTER_CUBIC)
    M = np.float32([[1,0,-trn*w/2],[0,1,-trn*h/2]])
    dst = cv2.warpAffine(res,M,(w,h))
    return dst


################################################################
############## description of setting parameter ################
################################################################
# (default setting get from lane detection)
# shape : tuple of height and width of given img
# cross : form of [x1,y1,x2,y2] which is (x1,y1) and (x2,y2) 
#              are end points of crosswalk found in this image
# center : central lane found in this image
# side : side lane found in this image
# 
# (info needed for calibration transform)
# needed for transform that calibrate some skewing in this view
# deg : degree of rotated angle of this view
# scale : scaling need for eliminating black img in rotated img
# rotM : rotation matrix that is needed for rotating 'deg'
#
# (info needed for perspective transform)
# needed for accurate distance calculation in perspective view
# vanP : vanishing point (point that central and side lane meet)
# previousRegion : region that will be perspective transformed
# afterRegion : region that will be in transformed coordinate
#
# (info needed for position detection)
# position detection will mainly held in perspective transformed 
# 2D coordinate. 
# grid : basis of scale in 2D coordinate
# trn_cross : position of crosswalk in 2D coordinate
# trn_bump : length of bump (unused)

################################################################


# function : calibrate this view of video by making initial setting parameter
#            rotate and scale up this image and calculate scale setting param
# INPUT : img or frame of this view
# OUTPUT : calibrated setting parameter
def calibration(img,res=None):

    if (not res):
        found = find(img)
        param = setting(found)
    else:
        param = res

    h = img.shape[0]
    w = img.shape[1]

    # scale factor
    degree = param['deg']
    angle = math.atan2(degree,1.0)
    cos = math.cos(angle)
    sin = math.sin(angle)
    scale = cos+w/h*sin

    # rotate first
    origin = [[w/2,h/2]]
    new_axis1 = rotL(param['center'],degree,origin)
    new_axis2 = rotL(param['side'],degree,origin)
    new_cross = rotL(param['cross'],degree,origin)
    new_vanP = rotP([param['vanP']],degree,origin)[0]

    # scale up next
    new_axis1 = scaL(new_axis1,scale,origin)
    new_axis2 = scaL(new_axis2,scale,origin)
    new_cross = scaL(new_cross,scale,origin)
    new_vanP = scaP([new_vanP],scale,origin)[0]

    lst1=[0,0,0,0]
    cnt1=0
    for el in param['prevRegion']:
        lst1[cnt1] = rotP([el],degree,origin)
        lst1[cnt1] = scaP(lst1[cnt1],scale,origin)[0]
        cnt1+=1

    new_res = dict()
    new_res = param
    new_res['afterRegion'] = param['afterRegion']    
    new_res['prevRegion'] = lst1
    new_res['persM'] = cv2.getPerspectiveTransform(np.float32(new_res['prevRegion']), np.float32(new_res['afterRegion'])) 
    new_res['scale'] = scale
    new_res['center'] = new_axis1
    new_res['side'] = new_axis2
    new_res['cross'] = new_cross
    new_res['vanP'] = new_vanP
    new_res['deg'] = degree

    rows = img.shape[1]
    cols = img.shape[0]
    angle = math.atan2(param['deg'],1.0)
    angle = angle/math.pi*180
    M = cv2.getRotationMatrix2D((rows/2,cols/2),angle,1)
    new_res['rotM'] = M
    rimg = transform(img,new_res)

    #cv2.imshow('rot_sca image', rimg)

    new_res = calc_setting(rimg,new_res)
    #print(new_res)

    return new_res


# function : transform calibrated image by rotating and scaling
# INPUT : img or frame of this view, calibrated setting (param)
# OUTPUT : calibrated(transformed) image 
# transform input image(or vidieo) regarding to input param
# param will include rotating angle, central, side line, scale factor
def transform(img,param):
    # img is form of numpy array of (B,G,R) pixel value
    # degree is form of tan(angle), same as slope
    dst1 = rotation(img,param)
    #cv2.imshow('a',dst1)
    dst2 = scaling(dst1,param)
    #cv2.imshow('b',dst2)
    return dst2

    