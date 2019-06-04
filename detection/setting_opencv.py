# @Author  : JaeminBest
# @File    : detection/setting_opencv.py
# @IDE: Microsoft Visual Studio Code

import cv2
import imutils
import numpy as np
import math
#from detection.calibration import transform


# needed for cropping image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# draw line to img, debug function
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
            cv2.line(line_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
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
# -1 for not on the same line, return 1 for same line
def onsameline(line1,line2, slope_thld, dist_thld):
    # cv2.line type line1, line2
    # check for slope difference
    if (math.fabs(slope(line1)-slope(line2))>slope_thld):
        return 0
    
    for x1,y1,x2,y2 in line2:
        # check for possibility of estimation of furthest point to be on same line
        if ((distance(line1,x1,y1)>dist_thld) or (distance(line1, x2, y2)>dist_thld)):
            return 0
    return 1   
 

# test if line2 can be extended to line1
# 0 for not on the parallel line, return 1 for parallel line
def isparallel(line1,line2, slope_thld):
    # cv2.line type line1, line2
    # check for slope difference
    if slope(line1) is None:
        return 0
    if slope(line2) is None:
        return 0
    if (math.fabs(slope(line1)-slope(line2))>slope_thld):
        #print("filtered by slope parallelism")
        return 0
    return 1   


# get intersection
def get_intersect(line1,line2):
    if isparallel(line1,line2,0):
        return None

    for a,b,c,d in line1:
        x1 = a
        y1 = b
        x2 = c
        y2 = d
    for a,b,c,d in line2:
        x3 = a
        y3 = b
        x4 = c
        y4 = d

    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [int(px), int(py)]


# new class for line sort
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


# detection of crosswalk
def detect_crosswalk(image,region, threshold = 50, slope_thld=0.005, dist_thld=0.01):
    # region is list of 2 tuple (x,y) i.e. rectangular region described by diagonal 2 points
    # region of interest using region selected
    region_of_interest_vertices = [
        (region[0][0], region[0][1]),
        (region[0][0], region[1][1]),
        (region[1][0], region[1][1]),
        (region[1][0], region[0][1])
    ]
    #cv2.imshow('cross_org_image', image)
    img = image.copy()

    # color detection
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_mask = np.array([30,0,200])
    upper_mask = np.array([224,30,230])
    mask = cv2.inRange(hsv, lower_mask, upper_mask)
    
    # Bitwise-AND mask and original image
    #cv2.imshow('cross_original', img)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= mask)
    temp_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('cross_masked_gray', temp_img)
    
    edges = cv2.Canny(temp_img, 50, 150, apertureSize=3)

    # crop
    cropped_image = region_of_interest(
        edges,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )
    #cv2.imshow('cross_mask_crop', cropped_image)

    # line detection
    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=100,
        lines=np.array([]),
        minLineLength=threshold,
        maxLineGap=50
    )

    if (lines is None):
        return None

    #lines1 = cv2.HoughLines(cropped_image,1,np.pi/180,100)

    #line_temp_image = draw_lines(
    #    img,
    #    lines,
    #    color = [0,0,255],
    #    thickness=5,
    #)
    #cv2.imshow('cross_houghP',line_temp_image)

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
    #cv2.imshow('cross_hough',res)
    
    lines_sort = []
    # sort lines by slope and gather using class el_line
    for line in lines:
        sl = slope(line)
        if (sl is None):
            continue
        # zero division set by -1 therefore no count!
        if ((sl < 0)):
            continue
            
        if (not lines_sort):
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort.append(temp)

        else:
            sort_flag = 0
            for el in lines_sort:
                if (onsameline(el.asymptote,line,slope_thld,dist_thld)==1):
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort.append(temp)
    
    #for el in lines_sort:
    #    print(el)

    # exception : no line matching
    if (not lines_sort):
        return None

    
    crosswalk = lines_sort[0]
    for el in lines_sort:
        if (len(crosswalk.line_x)<len(el.line_x)):
            crosswalk = el
    crosswalk = crosswalk.asymptote

    # draw line
    line_image = draw_lines(
        img,
        [ crosswalk ],
        thickness=5,
    )
    #cv2.imshow('img_cross',line_image)
    #cv2.waitKey()
    
    return crosswalk


# detection of central line (yellow one)
def detect_central(image,region, threshold = 150, slope_thld=0.05, dist_thld=0.5):
    # region is list of tuple (x,y)
    # region of interest using region selected
    region_of_interest_vertices = [
        (region[0][0], region[0][1]),
        (region[0][0], region[1][1]),
        (region[1][0], region[1][1]),
        (region[1][0], region[0][1])
    ]

    img = image.copy()

    # central line detection
    
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
    lines1 = cv2.HoughLinesP(
        morp3,
        rho=6,
        theta=np.pi / 60,
        threshold=200,
        lines=np.array([]),
        minLineLength=threshold,
        maxLineGap=25
    )

    if (lines1 is None):
        return None

    #lines1 = cv2.HoughLines(morp3,1,np.pi/180,100)

    line_temp_image = draw_lines(
        img,
        lines1,
        color = [0,0,255],
        thickness=5,
    )
    #cv2.imshow('img_hough',line_temp_image)

    lines_sort1 = []
    # sort lines by slope and gather using class el_line
    for line in lines1:
        sl = slope(line)
        if (sl is None):
            continue
        # zero division set by -1 therefore no count!
        if ((sl > 0) or (sl<=-0.2)):
            continue
            
        if (not lines_sort1):
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort1.append(temp)

        else:
            sort_flag = 0
            for el in lines_sort1:
                if (onsameline(el.asymptote,line,slope_thld,dist_thld)==1):
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort1.append(temp)

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    lines_sort1c = lines_sort1.copy()
    for el in lines_sort1:
        if ((slope(el.asymptote)>0) or (slope(el.asymptote)<=-0.2)):
            lines_sort1c.remove(el)
            continue
        temp_line = el.asymptote
        for x1,y1,x2,y2 in temp_line:
            if (((x1<img.shape[1]) and (x1>0)) or ((x2<img.shape[1]) and (x2>0))):
                lines_sort1c.remove(el)
            elif ((x1<0) or (x2>0)):
                lines_sort1c.remove(el)

    central = lines_sort1c[0].asymptote

    for el in lines_sort1c:
        if (slope(central)>slope(el.asymptote)):
            central = el.asymptote

    # side lane detection
    # extracting only road
    # using L*a*b*lst = []
    lab2 = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    lower_mask3 = np.array([0,0,0])
    upper_mask3 = np.array([155,140,130])
    mask3 = cv2.inRange(lab2, lower_mask3, upper_mask3)
    res3 = cv2.bitwise_and(img,img, mask= mask3)
    img3 = cv2.cvtColor(res3,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('central_Lab_road filter',img3)

    # define range of blue color in HSV
    kernel21 = np.ones((5, 5), np.uint8)
    morp21 = cv2.morphologyEx(img3, cv2.MORPH_OPEN, kernel21)
    #cv2.imshow('central_extract_open', morp21)
    kernel22 = np.ones((5, 5), np.uint8)
    morp22 = cv2.morphologyEx(morp21, cv2.MORPH_CLOSE, kernel22)
    #cv2.imshow('central_extract_open_close', morp22)
    kernel23 = np.ones((5, 5), np.uint8)
    morp23 = cv2.erode(morp22, kernel23, iterations = 1)
    #cv2.imshow('central_extract_open_close_erode', morp23)

    edges = cv2.Canny(morp23, 50, 150, apertureSize=3)
    #cv2.imshow('central_extract_road_edge', edges)

    # line detection
    lines2 = cv2.HoughLinesP(
        edges,
        rho=6,
        theta=np.pi / 60,
        threshold=200,
        lines=np.array([]),
        minLineLength=threshold,
        maxLineGap=25
    )

    if (lines2 is None):
        return None

    #lines1 = cv2.HoughLines(morp3,1,np.pi/180,100)

    line_temp_image = draw_lines(
        img,
        lines2,
        color = [0,0,255],
        thickness=5,
    )
    #cv2.imshow('img_hough',line_temp_image)

    lines_sort2 = []
    # sort lines by slope and gather using class el_line
    for line in lines2:
        sl = slope(line)
        if (sl is None):
            continue
        # zero division set by -1 therefore no count!
        if ((sl > 0) or (sl<=-0.2)):
            continue
            
        if (not lines_sort2):
            temp = el_line()
            temp.add(line)
            temp.estimate(img)
            lines_sort2.append(temp)

        else:
            sort_flag = 0
            for el in lines_sort2:
                if (onsameline(el.asymptote,line,slope_thld,dist_thld)==1):
                    el.add(line)
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                temp = el_line()
                temp.add(line)
                temp.estimate(img)
                lines_sort2.append(temp)

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    lines_sort2c = lines_sort2.copy()
    for el in lines_sort2:
        if ((slope(el.asymptote)>0) or (slope(el.asymptote)<=-0.2)):
            lines_sort2c.remove(el)
            continue
        temp_line = el.asymptote
        for x1,y1,x2,y2 in temp_line:
            if (((x1<img.shape[1]) and (x1>0)) or ((x2<img.shape[1]) and (x2>0))):
                lines_sort2c.remove(el)
            elif ((x1<0) or (x2>0)):
                lines_sort2c.remove(el)

    side = lines_sort2c[0].asymptote

    for el in lines_sort2c:
        if (slope(side)<slope(el.asymptote)):
            side = el.asymptote

    line_image = draw_lines(
        img,
        [[
            central[0],
            side[0],
        ]],
        thickness=5,
    )

    #cv2.imshow('img_central',line_image)
    #cv2.waitKey()
    return [side, central]


# detection of bump
# using this, we will set scale using already-known length
# match length with known value=360cm (it is already-set by LAW)
# RETURN : sizeof bump, position of crosswalk, newly transformed coordinate image
def detect_bump(image, param, threshold=25, dist_thld=0.0001):
    # input image should be perspective transformed one and including at least one bump
    img = image.copy()
    h = img.shape[0]
    w = img.shape[1]
    #cv2.imshow('org', img)

    # detection of crosswalk at warp perspective image
    lower_mask = np.array([0,0,240])
    upper_mask = np.array([255,255,255])
    mask = cv2.inRange(img, lower_mask, upper_mask)
    res = cv2.bitwise_and(img,img, mask= mask)
    #cv2.imshow('bump_mask',res)
    temp_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    lines = cv2.HoughLinesP(
        temp_img,
        rho=6,
        theta=np.pi / 60,
        threshold=150,
        lines=np.array([]),
        minLineLength=10,
        maxLineGap=25
    )

    # determination of position of crosswalk
    cross = 0
    maxline = lines[0]
    for line in lines:
        if slope(line) is None:
            cross = line[0][0]
            break
        if slope(line)>slope(maxline):
            maxline = line
    if (not cross):
        cross = maxline[0][0]

    param['trn_cross'] = cross

    # detection of bump
    luv = cv2.cvtColor(img,cv2.COLOR_BGR2Luv)
    lower_mask = np.array([150,0,0])
    upper_mask = np.array([255,255,255])
    mask = cv2.inRange(luv, lower_mask, upper_mask)
    res = cv2.bitwise_and(img,img, mask= mask)
    #cv2.imshow('bump_mask',res)
    temp_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    region_of_interest_vertices = [
        (cross+20, 0),
        (w, 0),
        (w, h),
        (cross+20, h)
    ]

    # bump
    cropped_image = region_of_interest(
        temp_img,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    #cv2.imshow('bump_mask',cropped_image)

    #edges = cv2.Canny(morp2, 50, 150, apertureSize=3)

    blank_img = np.zeros((h,w,3),np.uint8)    
    line_img = draw_lines(
        blank_img,
        [
            [[0,h/2,w,h/2],]
        ],
        color = [0,0,255],
        thickness=5,
    )
    line_img = cv2.cvtColor(line_img,cv2.COLOR_BGR2GRAY)

    # bump1
    new_img = np.where(cropped_image>0, line_img, 0)

    #cv2.imshow('bump detected',new_img)
    #cv2.waitKey()
    # line detection
    lines_bump = cv2.HoughLinesP(
        new_img,
        rho=6,
        theta=np.pi / 60,
        threshold=100,
        lines=np.array([]),
        minLineLength=3,
        maxLineGap=25
    )
    line_img_af = draw_lines(
        img,
        lines_bump,
        color = [0,0,255],
        thickness=5,
    )
    #cv2.imshow('after', line_img_af)
    #cv2.waitKey()
    dist = 0
    for line in lines_bump:
        for x1,y1,x2,y2 in line:
            p1 = np.array((x1,y1))
            p2 = np.array((x2,y2))
            temp = np.linalg.norm(p1-p2)
        if temp>dist:
            dist=temp
    
    #cv2.imshow('bump1_1',line_temp_image1)
    #cv2.imshow('bump2_1',line_temp_image2)
    #print(len(lines2))
    #print(len(lines1))
    param['trn_bump'] = dist
    return param

# function find crosswalk, central lane, side lane in given image
# INPUT : input img or frame, threshold for detection (slope_thld, dist_thld)
# OUTPUT : list of (height,width), crosswalk, central lane, side lane
def find(img, crslope_thld = 0.005, crdist_thld=0.01, cnslope_thld=0.05, cndist_thld=0.5):
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

    # central line detection
    lst = []
    res_central = detect_central(img,region1, threshold=150, slope_thld=cnslope_thld, dist_thld=cndist_thld)
    max_line = res_central[0]
    min_line = res_central[1]
    lst.append(max_line[0])
    lst.append(min_line[0])
    lst = []
    for i in range(100,201,10):
        #print(i)
        res_central = detect_central(img,region1,threshold=i, slope_thld=cnslope_thld, dist_thld=cndist_thld)
        if (res_central is None):
            continue
        #print(res_central)
        if (slope(min_line)>slope(res_central[1])):
            min_line = res_central[1]
        if (slope(max_line)<slope(res_central[0])):
            max_line = res_central[0]
        lst.append(res_central[1][0])
        lst.append(res_central[0][0])

    if (not lst):
        return None

    # crosswalk detection
    clst = []
    sidx=50
    step = 10
    for i in range(sidx,201,step):
        #print(i)
        res_cross = detect_crosswalk(img,region2,threshold=i, slope_thld=crslope_thld, dist_thld=crdist_thld)
        if (res_cross is None):
            continue
        #print(res_cross)        
        clst.append(res_cross)
    
    if (not clst):
        return None

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
                if (onsameline(el.asymptote,line,crslope_thld,crdist_thld)==1):
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

    if (not dlst):
        return None

    crosswalk = dlst[0]
    for el in dlst:
        if (len(crosswalk.line_x)<len(el.line_x)):
            crosswalk = el
    crosswalk = crosswalk.asymptote

    return ((height,width),crosswalk,min_line,max_line)


# function : calculate calibration setting such as rotated angle, or vanishing view etc.. 
#           for detail description of calibration setting, go to detection/calibration.py
# INPUT : shape(tuple of height and width), crosswalk, central lane, side lane
# OUTPUT : initial calibration setting (param) 
def setting(shape,crosswalk,center,side):
    height = shape[0]
    width = shape[1]
    # RETURN VALUE : json
    # deg : slope of rotation
    # cross : line of cross
    # axis1 : line of leftmost lane
    # axis2 : line of central lane
    res = dict()
    res['shape'] = (height,width)
    res['deg'] = slope(crosswalk)
    res['cross'] = crosswalk
    res['center'] = center
    res['side'] = side
    #print(min_line)
    #print(max_line)
    res['vanP'] = get_intersect(center,side)


    central = center
    x,y = res['vanP']
    w = width
    h = height

    # change warp perspective setting
    q1 = [1200,10]
    q2 = [10,10]
    q3 = [1200,50]
    q4 = [10,50]

    pts2 = np.float32([q1,q2,q3,q4])

    slop = slope(crosswalk)
    # bottom boundary
    slop1 = slope(side)
    slop2 = slope(central)
    x1 = 0
    y1 = -slop2*central[0][0]
    x2 = w
    y2 = y1+slop*x2
    bot = [[x1,y1,x2,y2]]

    if ((x<0) or (x>w) or (y<0) or (y>h)):
        # top boundary
        print(1)
        x3 = 0
        y3= -slop2*central[0][0]
        x4 = w
        y4 = slop*x4+y3
        top = [[x3,y3,x4,y4]]
        p1 = get_intersect(top,side)
        p2 = get_intersect(bot,side)
        p3 = get_intersect(top,central)
        p4 = get_intersect(bot,central)
        pts1 = np.array([p1,p2,p3,p4], dtype=np.float32)
    else: 
        # top boundary
        #print(2)
        y3=y+10
        x3 = x
        x4 = 0
        y4 = (y3-slop*x3)
        top = [[x3,y3,x4,y4]]
        p1 = get_intersect(top,side)
        p2 = get_intersect(bot,side)
        p3 = get_intersect(top,central)
        p4 = get_intersect(bot,central)
        pts1 = np.float32([p1,p2,p3,p4])
        #print(pts1)
    # draw line
    line_image = draw_lines(
        img,   
        [[
            top[0],
            bot[0],
            center[0],
            side[0],
            crosswalk[0],
        ]],
        thickness=5,
    )

    M = cv2.getPerspectiveTransform(pts1, pts2)
    res['persM'] = M
    res['prevRegion'] = [p1,p2,p3,p4]
    res['afterRegion'] = [q1,q2,q3,q4]
    cv2.imshow('img_detected',cv2.resize(line_image,dsize=(1200,600)))
    return res


# select proper json object which is very frequently appeared
def selection(img, res_lst, crslope_thld = 0.005, crdist_thld=0.01, cnslope_thld=0.05, cndist_thld=0.5, sdslope_thld=0.05, sddist_thld=0.5):
    crlst = []
    cnlst = []
    sdlst = []

    for res in res_lst:
        # for center
        if (not cnlst):
            #print('new line')
            temp = el_line()
            temp.add(res['center'])
            temp.estimate(img)
            cnlst.append(temp)
        else:
            sort_flag = 0
            for el in cnlst:
                if (onsameline(el.asymptote,res['center'],cnslope_thld,cndist_thld)==1):
                    #print('line extension')
                    el.add(res['center'])
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                #print('new line')
                temp = el_line()
                temp.add(res['center'])
                temp.estimate(img)
                cnlst.append(temp)

        # for side lane
        if (not sdlst):
            #print('new line')
            temp = el_line()
            temp.add(res['side'])
            temp.estimate(img)
            sdlst.append(temp)
        else:
            sort_flag = 0
            for el in sdlst:
                if (onsameline(el.asymptote,res['side'],sdslope_thld,sddist_thld)==1):
                    #print('line extension')
                    el.add(res['side'])
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                #print('new line')
                temp = el_line()
                temp.add(res['side'])
                temp.estimate(img)
                sdlst.append(temp)
        
        # for crosswalk
        if (not crlst):
            #print('new line')
            temp = el_line()
            temp.add(res['cross'])
            temp.estimate(img)
            crlst.append(temp)
        else:
            sort_flag = 0
            for el in crlst:
                if (onsameline(el.asymptote,res['cross'],crslope_thld,crdist_thld)==1):
                    #print('line extension')
                    el.add(res['cross'])
                    el.estimate(img)
                    sort_flag = 1
            if (not sort_flag):
                #print('new line')
                temp = el_line()
                temp.add(res['cross'])
                temp.estimate(img)
                crlst.append(temp)

    center = cnlst[0]
    for el in cnlst:
        if (len(center.line_x)<len(el.line_x)):
            center = el
    center = center.asymptote

    side = sdlst[0]
    for el in cnlst:
        if (len(side.line_x)<len(el.line_x)):
            side = el
    side = side.asymptote

    crosswalk = crlst[0]
    for el in crlst:
        if (len(crosswalk.line_x)<len(el.line_x)):
            crosswalk = el
    crosswalk = crosswalk.asymptote

    return ((img.shape[0],img.shape[1]),crosswalk,center,side)


# function : add new key 'grid' which will be scale meter of distance calculation 
#             between obj and crosswalk. can be calculated by perspective transform 
#             and image detection of bump. bump has known length of 3.6m
#             not only 'grid'
# INPUT : input img(must be img used for calibration which is no cars, people), 
#         param(setting that pass thru setting(), must have initial key-value)
# OUTPUT : newly set parameter that added key of 'grid' and 'trn_cross' which is
#           scale and position of crosswalk in 2D coordinate
def calc_setting(img,param):
    # draw line
    line_image = draw_lines(
        img,   
        [[
            param['cross'][0],
        ]],
        color = [0,0,255],
        thickness=3,
    )
    #cv2.imshow('main_calc_setting', line_image)
    timg = cv2.warpPerspective(line_image, param['persM'], (param['afterRegion'][0][0]+10,param['afterRegion'][3][1]+10))
    #cv2.imshow('main_calc_pers', timg)
    #cv2.waitKey()
    # bump detection
    res = detect_bump(timg,param)
    res['grid'] = 3.6/param['trn_bump']
    return res


# function : construct coordinate of 2D and 3D of targeting lanes
#            need for display and debugging
# INPUT : calibration setting parameter
# OUTPUT : coordinate of 2D (cord2) and coordinate of 3D (cord3)
#           both are just blank image with just crosswalk, central, side lane drawn 
def construct_cord(param):
    h = param['shape'][0]
    w = param['shape'][1]

    # coordinate of 3D image
    blank_image3 = np.zeros((h,w,3), np.uint8)
    cord3 = draw_lines(
        blank_image3,   
        [[
            param['center'][0],
            param['side'][0],
        ]],
        color = [255,0,0],
        thickness=1,
    )
    cord3 = draw_lines(
        cord3,   
        [[
            param['cross'][0],
        ]],
        color = [0,0,255],
        thickness=1,
    )
    cv2.imshow('cord image_3D', cv2.resize(cord3,dsize=(1200,600)))

    # coordinate of 2D image
    blank_image2 = np.zeros((param['afterRegion'][3][1]+10,param['afterRegion'][0][0]+10,3), np.uint8)
    cord2 = draw_lines(
        blank_image2,   
        [[
            [param['afterRegion'][0][0],param['afterRegion'][0][1],param['afterRegion'][1][0],param['afterRegion'][1][1]],
            [param['afterRegion'][2][0],param['afterRegion'][2][1],param['afterRegion'][3][0],param['afterRegion'][3][1]],
        ]],
        color = [255,0,0],
        thickness=1,
    )
    cord2 = draw_lines(
        cord2,
        [
            [[param['trn_cross'],0.,param['trn_cross'],cord2.shape[0]],]
        ],
        color = [0,0,255],
        thickness=1,
    )
    cv2.imshow('cord image_2D', cord2)
    return [cord3,cord2]