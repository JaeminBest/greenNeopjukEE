# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from detection.setting_opencv import setting, construct_cord
from detection.calibration import calibration, transform
from detection.measure import position, speed
import cv2
import numpy as np
import os
from os.path import isfile, join

# initial setting for given angle
class sesion():
    def __init__(self):
        self.deg = None
        self.central=None
        self.side=None
        self.cross=None
        self.scale=None
        self.vanP = None
        self.perspM = None
        self.prevRegion = None
        self.afterRegion = None
        self.grid = None

    
    def insert(self,res):
        self.deg = res['deg']
        self.central=res['central'] 
        self.side=res['side']
        self.cross=res['cross']
        self.scale=res['scale']
        self.vanP = res['vanP']
        self.persM = res['persM']
        self.prevRegion = res['prevRegion']
        self.afterRegion = res['afterRegion']

    def json(self):
        res = dict()
        res['deg'] = self.deg
        res['central'] = self.central
        res['side'] = self.side
        res['cross'] = self.cross
        res['scale'] = self.scale
        res['vanP'] = self.vanP
        res['persM'] = self.persM
        res['prevRegion'] = self.prevRegion
        res['afterRegion'] = self.afterRegion
        res['grid'] = self.grid

        return res

    def calib(self, img):
        res = calibration(img, self.json())
        self.insert(res)

    def __repr__(self):
        return "u'< deg:{}, central: {}, side:{}, cross:{}, scale:{}, bump1:{}, bump2:{} >".format(self.deg,self.central,self.side, self.cross, self.scale, self.bump1, self.bump2)


# initial dir setting for debugging
rootdir = './detection/data/'
datadir = '1.jpg'

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
def calibRecord(frames):
    param=None
    return param


# detect all objs
# input is image
# RETURN : info about pos,speed,count of objs 
def detectRecord(input, param):
    result=None
    return result


# measure all data from video and then send it to server 
def main():
    # open
    img = open(rootdir,datadir)
    cv2.imshow('main_org',cv2.resize(img,dsize=(1200,600)))
    #cv2.waitKey()
    # calibration
    res = setting(img)
    print(res)
    new_res = calibration(img,res)
    print(new_res)

    # construct coordinate image
    cords = construct_cord(img, new_res)
    cord3 = cords[0]
    cord2 = cords[1]

    # transform
    rimg = transform(img,new_res)
    cv2.imshow('main_transformed',cv2.resize(rimg,dsize=(1200,600)))
    

    
    # yolo-net detection (collabo with YOLO)
    # INPUT : rimg (rotated, scaled image)
    # OUTPUT : json object of object class/pixel value of boundary
    ###########################

    ###########################


    # draw point to rimg (collabo with SUMO)
    # INPUT: json object
    # OUTPUT : position value basis is line of crosswalk
    # sub-OUTPUT : coordinate image (2D,3D)
    ##########################
    position([[1650,680],[1650,680]],new_res,cord3,cord2)
    ##########################

    # warp perspective
    timg = cv2.warpPerspective(rimg, new_res['persM'], (new_res['afterRegion'][0][0]+10,new_res['afterRegion'][3][1]+10))
    cv2.imshow('main_persmode',timg)

    # detect point in measure.py
    
    ############
    
    
    cv2.waitKey()


if __name__=='__main__':
    main()