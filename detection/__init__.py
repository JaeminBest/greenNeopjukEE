import cv2
import numpy as np
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

rootdir = './detection/data/'
datadir = '1.jpg'

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img

# return slope
def slope(line):
    # line is type of cv2.line
    for x1,y1,x2,y2 in line:
        if (x2==x1):
            return -1
        slope = (float)(y2-y1)/(float)(x2-x1)
        return slope

# calculate distance(norm distance) between line(assumed to be straight line) and point(x,y)
def distance(line,x,y):
    p3 = (x,y)
    for x1, y1, x2, y2 in line:
        p1 = (x1, y1)
        p2 = (x2, y2)
    d=np.cross(p2-p1,p3-p1)/norm(p2-p1)
    return d

# test if line2 can be extended to line1
# -1 for not on the same line, return 1 for same line and save new tuple in line_x,line_y list
def onsameline(line1,line2, slope_thld, dist_thld):
    # cv2.line type line1, line2
    
    # check whether line2 is reverse form or not
    reverse = 0
    if (line1[0]>line1[3]):
        reverse = 1

    # check for slope difference
    if (math.fabs(slope(line1)-slope(line2))>slope_thld):
        return -1
    
    for x1,y1,x2,y2 in line2:
        # check for possibility of estimation of furthest point to be on same line
        if ((distnace(line1,x1,y1)>dist_thld) and (distance(line1, x2, y2)>dist_thld)):
            return -1
        return 1   


# new class for line sort
# store metadata of detected line as set and asymptote
class el_line():
    def __init__(self):
        self.line_x = []
        self.line_y = []
        self.lines = []
        self.asymptote = []

    def add(self,line):
        self.lines.append(line)
        for x1, x2, y1, y2 in line:
            self.line_x.extend([x1, x2])
            self.line_y.extend([y1, y2])
    
    def estimate(self,img):
        poly_left = np.poly1d(np.polyfit(
            self.line_y,
            self.line_x,
            deg=1
        ))
        min_y = (int)(0)
        max_y = (int)(img.shape[0])
        left_x_start = (int)(poly_left(min_y))
        left_x_end = (int)(poly_left(max_y))
        self.asymptote = [left_x_start,min_y,left_x_end,max_y]


# calibration of crosswalk
def calib_crosswalk(img,region, threshold = 3, slope_thld=0.05, dist_thld=2, cnt_thld=1, sort_thld = 0.2):
    # region is list of 2 tuple (x,y) i.e. rectangular region described by diagonal 2 points
    # region of interest using region selected
    region_of_interest_vertices = [
        (region[0][0], region[0][1]),
        (region[0][0], region[1][1]),
        (region[1][0], region[0][1]),
        (region[1][0], region[1][1])
    ]

    # color detection
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_mask = np.array([0,0,165])
    upper_mask = np.array([255,20,230])
    mask = cv2.inRange(hsv, lower_mask, upper_mask)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    temp_img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    gray_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200, apertureSize=threshold)
    
    # crop
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    # line detection
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    lines_sort = []

    # sort lines by slope and gather using class el_line
    for line in lines:
        sl = slope(line)
        # zero division set by -1 therefore no count!
        if ((sl > 0) or (sl <= -1)):
            continue

        if (not lines_sort):
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort.append(temp)
        
        else:
            sort_flag = 0
            for el in lines_sort:
                if onsameline(el.asymptote,lien,slope_thld,dist_thld):
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort.append(temp)

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    avg_slope = 0
    tot_count = 0
    for el in lines_sort:
        if (len(el.line_x)<cnt_thld):
            lines_sort.remove(el)
        else:
            avg_slope += slope(el.asymptote)
            tot_count += 1
    avg_slope = avg_slope/tot_count
    
    # delete lines_sort based on sort threshold
    # which means delete lines of thought to be noise slope
    for el in lines_sort:
        if (math.fabs(slope(el.asymptote)-avg_slope)>=sort_thld):
            lines_sort.remove(el)

    left_line = lines_sort[0].asymptote
    right_line = lines_sort[0].asymptote

    for el in lines_sort:
        #if (len(el.line_x)<cnt_thld):
        if (slope(left_line)>slope(el.asymptote)): 
            left_line = el.asymptote
        if (slope(right_line)<slope(el.asymptote)):
            right_line = el.asymptote

    line_image = draw_lines(
        img,
        [[
            left_line,
            right_line,
        ]],
        thickness=5,
    )

    cv2.imshow('img1',img)

    return [left_line, right_line]

# calibration of central line (yellow one)
def calib_central(img,region, threshold = 3, slope_thld=0.05, dist_thld=2, cnt_thld=1, sort_thld = 0.2):
    # region is list of tuple (x,y)
    # region of interest using region selected
    region_of_interest_vertices = [
        (region[0][0], region[0][1]),
        (region[0][0], region[1][1]),
        (region[1][0], region[0][1]),
        (region[1][0], region[1][1])
    ]

    # color detection
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_mask = np.array([11,50,50])
    upper_mask = np.array([30,255,255])
    mask = cv2.inRange(hsv, lower_mask, upper_mask)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    temp_img = cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    gray_image = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    cannyed_image = cv2.Canny(gray_image, 100, 200, apertureSize=threshold)
    
    # crop)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    # line detection
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    lines_sort = []

    # sort lines by slope and gather using class el_line
    for line in lines:
        sl = slope(line)
        # zero division set by -1 therefore no count!
        if ((sl < 0) or (sl>=1)):
            continue
            
        if (not lines_sort):
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort.append(temp)

        else:
            sort_flag = 0
            for el in lines_sort:
                if onsameline(el.asymptote,line,slope_thld,dist_thld):
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort.append(temp)

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    avg_slope = 0
    tot_count = 0
    for el in lines_sort:
        if (len(el.line_x)<cnt_thld):
            lines_sort.remove(el)
        else:
            avg_slope += slope(el.asymptote)
            tot_count += 1
    avg_slope = avg_slope/tot_count
    
    # delete lines_sort based on sort threshold
    # which means delete lines of thought to be noise slope
    for el in lines_sort:
        if (math.fabs(slope(el.asymptote)-avg_slope)>=sort_thld):
            lines_sort.remove(el)

    left_line = lines_sort[0].asymptote
    right_line = lines_sort[0].asymptote

    for el in lines_sort:
        #if (len(el.line_x)<cnt_thld):
        if (slope(left_line)>slope(el.asymptote)): 
            left_line = el.asymptote
        if (slope(right_line)<slope(el.asymptote)):
            right_line = el.asymptote

    line_image = draw_lines(
        img,
        [[
            left_line,
            right_line,
        ]],
        thickness=5,
    )
    cv2.imshow('img2',img)
    
    return [left_line, right_line]


# return json object having key as axis1, axis2, rotation value
def calibration(datadir=datadir, threshold=3):
    # file validationOLOR_BGR2HSV)
    
    dir = os.path.join(rootdir,datadir)
    if (os.path.isfile(dir)):
        img = cv2.imread(dir,cv2.IMREAD_COLOR)
    else:
        return -1

    # region of interest
    height = img.shape[0]
    width = img.shape[1]
    region1 = [
        (0, 0),
        (width, height / 2)
    ]
    region2 = [
        (width * 4/7, height * 3/7),
        (width * 6/7, height / 2)
    ]
    
    res = dict()
    res_central = calib_central(img,region1)
    res_cross = calib_crosswalk(img,region2)
    
    res_slope = (slope(res_central[0])+slope(res_central[1]))/2.0

    res['deg'] = slope
    res['axis1'] = (res_central[0].x1, res_central[0].y1, res_central[0].x2, res_central[0].y2)
    res['axis2'] = (res_central[1].x1, res_central[1].y1, res_central[1].x2, res_central[1].y2)
    
    returnstroyAllWindows()
    return res

if __name__=='__main__':
    calibration()