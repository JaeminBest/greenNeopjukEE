# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from setting_opencv import setting
from measure import *

# initial setting for given angle
class setting():
    def __init__(self):
        self.deg = None
        self.axis1=None
        self.axis2=None
        self.cross=None
        self.scale=None
    
    def update(self,deg,axis1,axis2,cross):
        self.deg = deg
        self.axis1=axis1
        self.axis2=axis2
        self.cross=cross
    
    def calc_scale(self,region,)


# make initial setting for total video
def calibration(rootdir=rootdir, datadir=datadir):
    res = setting(rootdir,datadir)
    
    
    return

# measure all data from 
def main():
    return

if __name__=='__main__':
    main()