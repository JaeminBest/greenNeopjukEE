# @Author  : JaeminBest
# @File    : detection/__init__.py
# @IDE: Microsoft Visual Studio Code

from setting_opencv import *
from calibration import *
from measure import *

# initial setting for given angle
class setting():
    def __init__(self):
        self.deg = None
        self.axis1=None
        self.axis2=None
        self.cross=None
    
    def update(self,deg,axis1,axis2,cross):
        self.deg = deg
        self.axis1=axis1
        self.axis2=axis2
        self.cross=cross


# make initial setting for total video
def calibration():
    

# measure all data from 
def main():


if __name__=='__main__':
    main()