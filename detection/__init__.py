# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from setting_opencv import setting
from calibration import calibration, transform
from measure import *
import cv2
import numpy as np
import os
from os.path import isfile, join

# initial setting for given angle
class setting():
    def __init__(self):
        self.deg = None
        self.axis1=None
        self.axis2=None
        self.bump1 = None
        self.bump2 = None
        self.cross=None
        self.scale=None
    
    def insert(self,res):
        self.deg = res['deg']
        self.central=res['central'] 
        self.side=res['side']
        self.cross=res['cross']
        self.bump1 = res['bump1']
        self.bump2 = res['bump2']

    def json(self):
        res = dict()
        res['deg'] = self.deg
        res['cross'] = self.cross
        res['central'] = self.central
        res['side'] = self.side
        res['bump1'] = self.bump1
        res['bump2'] = self.bump2
        res['scale'] = self.scale
        return res

    def calib(self, img):
        res = calibration(img, self.json())
        self.insert(res)

    def __repr__(self):
        return "u'< deg:{}, central: {}, side:{}, cross:{}, scale:{}, bump1:{}, bump2:{} >".format(self.deg,self.central,self.side, self.cross, self.scale, self.bump1, self.bump2)


# initial dir setting for debugging
rootdir = './data/'
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
    return param


# detect all objs
# input is image
# RETURN : info about pos,speed,count of objs 
def detectRecord(input, param):
    return result


# measure all data from video and then send it to server 
def main(mode):
    if (mode==0):

    if (mode==1):
    
    return

if __name__=='__main__':
    main()