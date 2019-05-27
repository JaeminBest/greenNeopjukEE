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
            return None
        slope = (float)(y2-y1)/(float)(x2-x1)
        return slope


# calculate distance(norm distance) between line(assumed to be straight line) and point(x,y)
def distance(line,x,y):
    p3 = np.array((x,y))
    for x1, y1, x2, y2 in line:
        p1 = np.array((x1, y1))
        p2 = np.array((x2, y2))
    d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    return d


# test if line2 can be extended to line1
# only need for calib_central
# -1 for not on the same line, return 1 for same line and save new tuple in line_x,line_y list
def onsameline(line1,line2, slope_thld, dist_thld):
    # cv2.line type line1, line2
    
    # check whether line2 is reverse form or not
    reverse = 0
    for x1,y1,x2,y2 in line1:
        if (x1>x2):
            reverse = 1

    # check for slope difference
    if (math.fabs(slope(line1)-slope(line2))>slope_thld):
        return -1
    
    for x1,y1,x2,y2 in line2:
        # check for possibility of estimation of furthest point to be on same line
        if ((distance(line1,x1,y1)>dist_thld) or (distance(line1, x2, y2)>dist_thld)):
            return -1
    return 1   


# test if line2 is similar line with line1 (which mean same slope value)
# only need for calib_crosswalk
# -1 for not on the same line, return 1 for same line and save new tuple in line_x,line_y list
def issimilarline(line, slope, slope_thld):
    # cv2.line type line1, line2
    
    # check whether line2 is reverse form or not
    reverse = 0
    for x1,y1,x2,y2 in line1:
        if (x1>x2):
            reverse = 1

    # check for slope difference
    if (math.fabs(slope(line)-slope)>slope_thld):
        return -1
    
    return 1   


# new class for line sort
# only need for calib_central
# store metadata of detected line as set and asymptote
class el_line():
    def __init__(self):
        self.line_x = []
        self.line_y = []
        self.lines = []
        self.asymptote = []
    
    def __repr__(self):
        return "< line_x : {}, line_y :{}, lines : {}, asymptote : {} >".format(self.line_x, self.line_y, self.lines, self.asymptote)

    def add(self,line):
        self.lines.append(line)
        for x1, y1, x2, y2 in line:
            if (x1 in self.line_x):
                self.line_y[self.line_x.index(x1)] = (self.line_y[self.line_x.index(x1)]+y1)/2
            else:
                self.line_x.extend([x1])
                self.line_y.extend([y1])
            if (x2 in self.line_x):
                self.line_y[self.line_x.index(x2)] = (self.line_y[self.line_x.index(x2)]+y2)/2
            else:
                self.line_x.extend([x2])
                self.line_y.extend([y2])

    def estimate(self,img):
        if (len(self.line_y)==2):
            y1 = self.line_y[0]
            y2 = self.line_y[1]
            if (y1==y2):
                self.asymptote = [[0,y1,img.shape[1],y1]]
                return

        poly_left = np.poly1d(np.polyfit(
            self.line_y,
            self.line_x,
            deg=1
        ))
        min_y = (int)(0)
        max_y = (int)(img.shape[0])
        left_x_start = (int)(poly_left(min_y))
        left_x_end = (int)(poly_left(max_y))
        self.asymptote = [[left_x_start,min_y,left_x_end,max_y]]
        return


# calibration of crosswalk
def calib_crosswalk(image,region, threshold = 100, slope_thld=0.005, dist_thld=0.01, cnt_thld=1, sort_thld = 0.2):
    # region is list of 2 tuple (x,y) i.e. rectangular region described by diagonal 2 points
    # region of interest using region selected
    region_of_interest_vertices = [
        (region[0][0], region[0][1]),
        (region[0][0], region[1][1]),
        (region[1][0], region[1][1]),
        (region[1][0], region[0][1])
    ]
    cv2.imshow('cross_org_image', image)
    img = image.copy()

    # color detection
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_mask = np.array([30,0,200])
    upper_mask = np.array([224,30,230])
    mask = cv2.inRange(hsv, lower_mask, upper_mask)
    
    # Bitwise-AND mask and original image
    cv2.imshow('0_cross', img)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    temp_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    cv2.imshow('1_cross', temp_img)
    #kernel1 = np.ones((3, 3), np.uint8)
    #morp1 = cv2.morphologyEx(temp_img, cv2.MORPH_OPEN, kernel1) 
    #cv2.imshow('2_cross', morp1)
    #kernel2 = np.ones((10, 10), np.uint8)
    #morp2 = cv2.morphologyEx(morp1, cv2.MORPH_CLOSE, kernel2) 
    #cv2.imshow('3_cross', morp2)
    #kernel3 = np.ones((5, 5), np.uint8)
    #morp3 = cv2.morphologyEx(morp2, cv2.MORPH_CLOSE, kernel3) 
    #cv2.imshow('4_cross', morp3)
    
    edges = cv2.Canny(temp_img, 50, 150, apertureSize=3)
    #cv2.imshow('4_cross', edges)

    # crop
    cropped_image = region_of_interest(
        edges,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
    cv2.imshow('4_cross', cropped_image)

    # line detection
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=100,
        lines=np.array([]),
        minLineLength=threshold/2,
        maxLineGap=50
    )

    if (lines is None):
        return None

    lines1 = cv2.HoughLines(cropped_image,1,np.pi/180,100)

    line_temp_image = draw_lines(
        img,
        lines,
        color = [0,0,255],
        thickness=5,
    )
    cv2.imshow('img_cross_hough',line_temp_image)

    for i in range(len(lines1)):
        for rho, theta in lines1[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + img.shape[1]*3*(-b))
            y1 = int(y0+img.shape[1]*3*(a))
            x2 = int(x0 - img.shape[1]*3*(-b))
            y2 = int(y0 -img.shape[1]*3*(a))
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    res = np.vstack((img.copy(), img))
    cv2.imshow('img_cross_houghP',res)
    

    lines_sort = []
    #print(lines)
    #print("======")
    # sort lines by slope and gather using class el_line
    #print('iteration start')
    for line in lines:
        #print('setting iteration')
        #print(line)
        sl = slope(line)
        if (sl is None):
            continue
        # zero division set by -1 therefore no count!
        if ((sl < 0)):
            #print('invalid slope')
            continue
            
        if (not lines_sort):
            #print('new line')
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort.append(temp)

        else:
            sort_flag = 0
            for el in lines_sort:
                if (onsameline(el.asymptote,line,slope_thld,dist_thld)==1):
                    #print('line extension')
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                #print('new line')
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort.append(temp)
    
    print("before filter")
    for el in lines_sort:
        print(el)

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    #avg_slope = 0
    #max_count = 0
    #for el in lines_sort:
    #    if ((slope(el.asymcv2.waitKey()ptote)<0)):
            #print('remove'cv2.waitKey())
    #        lines_sort.remcv2.waitKey()ove(el)
    #    else:cv2.waitKey()
    #        avg_slope += scv2.waitKey()lope(el.asymptote)
    #        tot_count += 1cv2.waitKey()
    
    # slope filtering by accv2.waitKey()tive region
    #for el in lines_sort:cv2.waitKey()
    #    temp_line = el.asymptote
    #    for x1,y1,x2,y2 in temp_line:
    #        if (((x1<img.shape[1]) and (x1>0)) or ((x2<img.shape[1]) and (x2>0))):
    #            lines_sort.remove(el)
    #        elif ((x1<0) or (x2>0)):
    #            lines_sort.remove(el)
    
    #if (tot_count!=0):
    #    avg_slope = avg_slope/tot_count
    #else:
    #    avg_slope = None
    
    # delete lines_sort based on sort threshold
    # which means delete lines of thought to be noise slope
    #for el in lines_sort:
    #    if (math.fabs(slope(el.asymptote)-avg_slope)>=sort_thld):
    #        lines_sort.remove(el)

    crosswalk = lines_sort[0]

    for el in lines_sort:
        if (len(crosswalk.line_x)<len(el.line_x)):
            crosswalk = el
    
    crosswalk = crosswalk.asymptote

    line_image = draw_lines(
        img,
        [ crosswalk ],
        thickness=5,
    )

    cv2.imshow('img_crossP',line_image)
    cv2.waitKey()
    
    return crosswalk

# calibration of central line (yellow one)
def calib_central(image,region, threshold = 150, slope_thld=0.05, dist_thld=0.5, cnt_thld=1, sort_thld = 0.1):
    # region is list of tuple (x,y)
    # region of interest using region selected
    region_of_interest_vertices = [
        (region[0][0], region[0][1]),
        (region[0][0], region[1][1]),
        (region[1][0], region[1][1]),
        (region[1][0], region[0][1])
    ]

    img = image.copy()

    # color detection
    # extracting central+crosswalk
    # using L*a*b*lst = []
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    lower_mask1 = np.array([110,132,134])
    upper_mask1 = np.array([220,255,255])
    mask1 = cv2.inRange(lab, lower_mask1, upper_mask1)
    res1 = cv2.bitwise_and(img,img, mask= mask1)
    img1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('central_Lab filter',img1)

    # extracting only crosswalk
    # using YCrCb
    ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    lower_mask2 = np.array([150,0,0])
    upper_mask2 = np.array([255,225,115])
    mask2 = cv2.inRange(ycrcb,lower_mask2, upper_mask2)
    res2 = cv2.bitwise_and(img,img, mask= mask2)
    img2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('central_YCrCb filter',img2)

    # subtraction, extracting ONLY central line
    new_img = np.subtract(img1,img2)
    new_img = np.where(img1-img2<0,0,img1-img2)
    #cv2.imshow('central_extract', new_img)

    # define range of blue color in HSV
    kernel1 = np.ones((5, 5), np.uint8)
    morp1 = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel1)
    #cv2.imshow('central_extract_open', morp1)
    kernel2 = np.ones((5, 5), np.uint8)
    morp2 = cv2.morphologyEx(morp1, cv2.MORPH_CLOSE, kernel2)
    #cv2.imshow('central_extract_open_close', morp2)
    kernel3 = np.ones((5, 5), np.uint8)
    morp3 = cv2.erode(morp2, kernel3, iterations = 1)
    #cv2.imshow('central_extract_open_close_erode', morp3)
    
    #edges = cv2.Canny(morp3, 100, 200, apertureSize=3)
    #cv2.imshow('central_final edge', edges)
    # line detection
    lines = cv2.HoughLinesP(
        morp3,
        rho=6,
        theta=np.pi / 60,
        threshold=200,
        lines=np.array([]),
        minLineLength=threshold,
        maxLineGap=25
    )

    if (lines is None):
        return None

    #lines1 = cv2.HoughLines(morp3,1,np.pi/180,100)

    line_temp_image = draw_lines(
        img,
        lines,
        color = [0,0,255],
        thickness=5,
    )
    #cv2.imshow('img_hough',line_temp_image)

    #for i in range(len(lines1)):
    #    for rho, theta in lines1[i]:
    #        a = np.cos(theta)
    #        b = np.sin(theta)
    #        x0 = a*rho
    #        y0 = b*rho
    #        x1 = int(x0 + img.shape[1]*3*(-b))
    #        y1 = int(y0+img.shape[1]*3*(a))
    #        x2 = int(x0 - img.shape[1]*3*(-b))
    #        y2 = int(y0 -img.shape[1]*3*(a))

    #        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    #res = np.vstack((img.copy(), img))
    #cv2.imshow('img',res)



    lines_sort = []
    #print(lines)
    #print("======")
    # sort lines by slope and gather using class el_line
    #print('iteration start')
    for line in lines:
        #print('setting iteration')
        #print(line)
        sl = slope(line)
        if (sl is None):
            continue
        # zero division set by -1 therefore no count!
        if ((sl > 0) or (sl<=-0.2)):
            #print('invalid slope')
            continue
            
        if (not lines_sort):
            #print('new line')
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort.append(temp)

        else:
            sort_flag = 0
            for el in lines_sort:
                if (onsameline(el.asymptote,line,slope_thld,dist_thld)==1):
                    #print('line extension')
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                #print('new line')
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort.append(temp)
    
    #print("before filter")
    #for el in lines_sort:
        #print(el)

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    avg_slope = 0
    tot_count = 0
    for el in lines_sort:
        if ((slope(el.asymptote)>0) or (slope(el.asymptote)<=-0.2)):
            #print('remove')
            lines_sort.remove(el)
        else:
            avg_slope += slope(el.asymptote)
            tot_count += 1
    
    # slope filtering by active region
    for el in lines_sort:
        temp_line = el.asymptote
        for x1,y1,x2,y2 in temp_line:
            if (((x1<img.shape[1]) and (x1>0)) or ((x2<img.shape[1]) and (x2>0))):
                lines_sort.remove(el)
            elif ((x1<0) or (x2>0)):
                lines_sort.remove(el)
    
    #if (tot_count!=0):
    #    avg_slope = avg_slope/tot_count
    #else:
    #    avg_slope = None
    
    # delete lines_sort based on sort threshold
    # which means delete lines of thought to be noise slope
    #for el in lines_sort:
    #    if (math.fabs(slope(el.asymptote)-avg_slope)>=sort_thld):
    #        lines_sort.remove(el)

    left_line = lines_sort[0].asymptote
    right_line = lines_sort[0].asymptote

    for el in lines_sort:
        #if (len(el.line_x)<cnt_thld):
        if (slope(left_line)<slope(el.asymptote)): 
            left_line = el.asymptote
        if (slope(right_line)>slope(el.asymptote)):
            right_line = el.asymptote

    #print("after filter")
    #for el in lines_sort:
        #print(el)

    #line_image = draw_lines(
    #    img,
    #    [[
    #        left_line[0],
    #        right_line[0],
    #    ]],
    #    thickness=5,
    #)

    #cv2.imshow('img_central',line_image)
    
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
        (width / 2, height * 2/3),
        (width, height * 1/3)
    ]
    region2 = [
        (0, height),
        (width, height / 2)
    ]
    
    res = dict()

    # central line calibration
    lst = []
    res_central = calib_central(img,region1, threshold=150)
    max1 = res_central[0]
    min1 = res_central[1]
    lst.append(max1[0])
    lst.append(min1[0])
    lst = []
    for i in range(100,201,10):
        print(i)
        res_central = calib_central(img,region1,threshold=i)
        if (res_central is None):
            continue
        print(res_central)
        if (slope(min1)>slope(res_central[1])):
            min1 = res_central[1]
        if (slope(max1)<slope(res_central[0])):
            max1 = res_central[0]
        lst.append(res_central[1][0])
        lst.append(res_central[0][0])
        
    print(min1)
    print(max1)
    line_image = draw_lines(
        img,
        [[
            min1[0],
            max1[0]
        ]],
        thickness=5,
    )

    cv2.imshow('img_central',line_image)

    res_cross = calib_crosswalk(img,region2,threshold=100)
    return

    # crosswalk calibration
    clst = []
    sidx=100
    step = 10
    for i in range(sidx,201,step):
        print(i)
        res_cross = calib_crosswalk(img,region2,threshold=i)
        if (res_cross is None):
            continue
        print(res_cross)        
        clst.append(res_cross)
    
    print(clst)
    
    dlst=[]
    for line in clst:
        #print('setting iteration')
        #print(line)
        if (not dlst):
            #print('new line')
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            dlst.append(temp)

        else:
            sort_flag = 0
            for el in dlst:
                if (onsameline(el.asymptote,line,slope_thld,dist_thld)==1):
                    #print('line extension')
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                #print('new line')
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                dlst.append(temp)

    crosswalk = dlst[0]
    for el in dlst:
        if (len(crosswalk.len_x)<len(el.len_x)):
            crosswalk = el

    crosswalk = crosswalk.asymptote

    line_image = draw_lines(
        img,
        [crosswalk],
        thickness=5,
    )

    cv2.imshow('img_cross',line_image)
    
    #res_slope = (slope(res_cross[0])+slope(res_cross[1]))/2.0

    res['deg'] = slope(crosswalk)
    res['axis1'] = min1
    res['axis2'] = max1
    return res

if __name__=='__main__':
    calibration()