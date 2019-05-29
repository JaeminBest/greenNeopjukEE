# @Author  : JaeminBest
# @File    : detection/calibration.py
# @IDE: Microsoft Visual Studio Code

import cv2
import numpy as np
import math
from setting_opencv import setting

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

    return [[qx, qy]]


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
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return [[qx, qy]]


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


# make initial setting for total video
# REUTRN VALUE : newly calibrated param
# newly calculate setting param for calibration
def calibration(img, param):

    param = setting(img)
    h,w = img.shape

    # scale factor
    degree = param['degree']
    angle = math.atan2(degree,1.0)
    cos = math.cos(angle)
    sin = math.sin(angle)
    scale = w/h*cos+h/w*sin

    # rotate first
    origin = [[h/2,w/2]]
    new_axis1 = rotL(param['center'],degree,origin)
    new_axis2 = rotL(param['side'],degree,origin)
    new_cross = rotL(param['cross'],degree,origin)

    # scale up next
    new_bump1 = scale*param['bump1']
    new_bump2 = scale*param['bump2']
    new_axis1 = scaL(new_axis1,scale,origin)
    new_axis2 = scaL(new_axis2,scale,origin)
    new_cross = scaL(new_cross,scale,origin)

    new_res = dict()
    new_res['scale'] = scale
    new_res['center'] = new_axis1
    new_res['side'] = new_axis2
    new_res['cross'] = new_cross
    new_res['bump1'] = new_bump1
    new_res['bump2'] = new_bump2
    
    return new_res


# transform input image(or vidieo) regarding to input param
# param will include rotating angle, central, side line, scale factor
# REUTRN VALUE : transformed image and param
def transform(img,scale,degree):
    # img is form of numpy array of (B,G,R) pixel value
    # degree is form of tan(angle), same as slope
    angle = math.atan2(degree,1.0)
    cols,rows = img.shape
    rotAng = angle*180/math.pi
    rotM = cv2.getRotationMatrix2D((cols/2,rows/2),rotAng,1)
    dst = cv2.warpAffine(img,rotM,(cols,rows))
    res = cv2.resize(dst,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    return res
