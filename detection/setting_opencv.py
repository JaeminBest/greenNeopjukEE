# @Author  : JaeminBest
# @File    : detection/setting_opencv.py
# @IDE: Microsoft Visual Studio Code

import cv2
import imutils
import numpy as np
import os
from os.path import isfile, join
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# initial dir setting for debugging
rootdir = './data/'
datadir = '1.jpg'

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
# -1 for not on the same line, return 1 for same line
def onsameline(line1,line2, slope_thld, dist_thld):
    # cv2.line type line1, line2
    # check for slope difference
    if (math.fabs(slope(line1)-slope(line2))>slope_thld):
        return -1
    
    for x1,y1,x2,y2 in line2:
        # check for possibility of estimation of furthest point to be on same line
        if ((distance(line1,x1,y1)>dist_thld) or (distance(line1, x2, y2)>dist_thld)):
            return -1
    return 1   
 

# test if line2 can be extended to line1
# -1 for not on the parallel line, return 1 for parallel line
def isparallel(line1,line2, slope_thld):
    # cv2.line type line1, line2
    # check for slope difference
    if slope(line1) is None:
        return 0
    if slope(line2) is None:
        return 0
    if (math.fabs(slope(line1)-slope(line2))>slope_thld):
        return 0
    return 1   


# find out contour approx is bump or not
# condition for bump : having 2 parallel line which are parallel to crosswalk
def isbump(approx, crosswalk,slope_thld):
    if len(approx)!=4:
        return -1
    print(approx)
    for p1,p2,p3,p4 in [approx]:
        print("p1:{},p2:{},p3:{},p4:{}".format(p1,p2,p3,p4))
        
        for x,y in p1:
            p1x=x
            p1y=y
        for x,y in p2:
            p2x=x
            p2y=y
        for x,y in p3:
            p3x=x
            p3y=y
        for x,y in p4:
            p4x=x
            p4y=y

        lst = []
        temp_line1 = [[p1x,p1y,p2x,p2y]]
        if (isparallel(temp_line1, crosswalk, slope_thld)):
            lst.append(1)
        temp_line2 = [[p2x,p2y,p3x,p3y]]
        if (isparallel(temp_line2, crosswalk, slope_thld)):
            lst.append(1)
        temp_line3 = [[p3x,p3y,p4x,p4y]]
        if (isparallel(temp_line3, crosswalk, slope_thld)):
            lst.append(1)
        temp_line4 = [[p4x,p4y,p1x,p1y]]
        if (isparallel(temp_line4, crosswalk, slope_thld)):
            lst.append(1)
        temp_line5 = [[p1x,p1y,p3x,p3y]]
        if (isparallel(temp_line5, crosswalk, slope_thld)):
            lst.append(1)
        temp_line6 = [[p2x,p2y,p4x,p4y]]
        if (isparallel(temp_line6, crosswalk, slope_thld)):
            lst.append(1)
        
        cnt = 0
        for el in lst:
            if (el==1):
                cnt+=1
        if (cnt>=2):
            print("pass")
            return 1
    print("fail")
    return -1


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
    # sort lines by slope and gather using class el_line
    for line in lines:
        sl = slope(line)
        if (sl is None):
            continue
        # zero division set by -1 therefore no count!
        if ((sl > 0) or (sl<=-0.2)):
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

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    for el in lines_sort:
        if ((slope(el.asymptote)>0) or (slope(el.asymptote)<=-0.2)):
            lines_sort.remove(el)
        else:
            continue
    
    # slope filtering by active region
    for el in lines_sort:
        temp_line = el.asymptote
        for x1,y1,x2,y2 in temp_line:
            if (((x1<img.shape[1]) and (x1>0)) or ((x2<img.shape[1]) and (x2>0))):
                lines_sort.remove(el)
            elif ((x1<0) or (x2>0)):
                lines_sort.remove(el)

    left_line = lines_sort[0].asymptote
    right_line = lines_sort[0].asymptote

    for el in lines_sort:
        if (slope(left_line)<slope(el.asymptote)): 
            left_line = el.asymptote
        if (slope(right_line)>slope(el.asymptote)):
            right_line = el.asymptote

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


# detection of bump
# using this, we will set scale using already-known length
# match length with known value=360cm (it is already-set by LAW)
def detect_bump(image, region1, region2, crosswalk, central1, central2, threshold=50, slope_thld=0.005, dist_thld=0.01):
    img = image.copy()

    # region of interest_vertices
    region_of_interest_vertices1 = [
        (region1[0][0], region1[0][1]),
        (region1[0][0], region1[1][1]),
        (region1[1][0], region1[1][1]),
        (region1[1][0], region1[0][1])
    ]
    region_of_interest_vertices2 = [
        (region2[0][0], region2[0][1]),
        (region2[0][0], region2[1][1]),
        (region2[1][0], region2[1][1]),
        (region2[1][0], region2[0][1])
    ]

    # edge image
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    lower_mask = np.array([140,130,130])
    upper_mask = np.array([200,255,170])
    mask = cv2.inRange(lab, lower_mask, upper_mask)
    res = cv2.bitwise_and(img,img, mask= mask)
    #cv2.imshow('bump_mask',res)
    temp_img = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    kernel1 = np.ones((3, 3), np.uint8)
    morp1 = cv2.morphologyEx(temp_img, cv2.MORPH_OPEN, kernel1)
    #cv2.imshow('bump_mask_open',morp1)
    kernel2 = np.ones((3, 3), np.uint8)
    morp2 = cv2.morphologyEx(morp1, cv2.MORPH_CLOSE, kernel2)
    #cv2.imshow('bump_mask_open_close',morp2)

    edges = cv2.Canny(morp2, 50, 150, apertureSize=3)

    # bump1
    cropped_image1 = region_of_interest(
        edges,
        np.array(
            [region_of_interest_vertices1],
            np.int32
        ),
    )
    cv2.imshow('bump_bump1_crop',cropped_image1)
    # line detection
    lines1 = cv2.HoughLinesP(
        cropped_image1,
        rho=6,
        theta=np.pi / 60,
        threshold=200,
        lines=np.array([]),
        minLineLength=threshold,
        maxLineGap=25
    )

    if (lines1 is None):
        return None

    lines_sort = []
    # sort lines by slope and gather using class el_line
    for line in lines1:
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

    # delete lines_sort based on count threshold
    # which means delete less-detected slope of lines
    for el in lines_sort:
        if ((slope(el.asymptote)>0) or (slope(el.asymptote)<=-0.2)):
            lines_sort.remove(el)
        else:
            continue
    
    # slope filtering by active region
    for el in lines_sort:
        temp_line = el.asymptote
        for x1,y1,x2,y2 in temp_line:
            if (((x1<img.shape[1]) and (x1>0)) or ((x2<img.shape[1]) and (x2>0))):
                lines_sort.remove(el)

    min_line = lines_sort[0].asymptote
    max_line = lines_sort[0].asymptote
    
    for el in lines_sort:
        if (slope(left_line)<slope(el.asymptote)): 
            left_line = el.asymptote
        if (slope(right_line)>slope(el.asymptote)):
            right_line = el.asymptote

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

    # bump2
    cropped_image2 = region_of_interest(
        edges,
        np.array(
            [region_of_interest_vertices2],
            np.int32
        ),
    )
    cv2.imshow('bump_bump2_crop',cropped_image2)
    #cv2.waitKey()
    threshold2 = cv2.threshold(cropped_image2, 200, 255, cv2.THRESH_BINARY)[1]
    contours2 = cv2.findContours(threshold2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = imutils.grab_contours(contours2)
    
    # bump detection
    font = cv2.FONT_HERSHEY_COMPLEX

    bump1_lst=[]
    for cnt in contours1:
        print('itr for contours1')
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx)==4:
            # bump detected
            if isbump(approx, crosswalk,slope_thld):
                bump1_lst.append(approx)
                cv2.putText(img, "Bump1", (x, y), font, 1, (0))
        else:
            continue

    bump2_lst=[]
    for cnt in contours2:
        print('itr for contours2')
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0), 5)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if len(approx)==4:
            # bump detected
            if isbump(approx, crosswalk,slope_thld):
                bump2_lst.append(approx)
                cv2.putText(img, "Bump2", (x, y), font, 1, (0))
        else:
            continue
    
    cv2.imshow("shapes", img)
    cv2.imshow("Threshold1", threshold1)
    cv2.imshow("Threshold2", threshold2)
    
    cv2.waitKey(0)
    return [bump1_lst, bump2_lst]



# return json object having key as axis1, axis2, rotation value
def setting(rootdir=rootdir, datadir=datadir, crslope_thld = 0.005, crdist_thld=0.01, cnslope_thld=0.05, cndist_thld=0.5):
    # file validation    
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
    region3 = [
        (width/2, height / 2),
        (width, height * 4/7)
    ]
    region4 = [
        (0, height * 4/7),
        (width, height * 6/7)
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

    crosswalk = dlst[0]
    for el in dlst:
        if (len(crosswalk.line_x)<len(el.line_x)):
            crosswalk = el
    crosswalk = crosswalk.asymptote

    # bump detection
    [bump1, bump2] = detect_bump(img,region3,region4,crosswalk,min_line, max_line)
    print(bump1)
    print(bump2)

    # RETURN VALUE : json
    # deg : slope of rotation
    # cross : line of cross
    # axis1 : line of leftmost lane
    # axis2 : line of central lane
    res = dict()
    res['deg'] = slope(crosswalk)
    res['cross'] = crosswalk
    res['axis1'] = min_line
    res['axis2'] = max_line


    # draw line
    line_image = draw_lines(
        img,
        [[
            min_line[0],
            max_line[0],
            crosswalk[0],
        ]],
        thickness=5,
    )

    cv2.imshow('img_detected',line_image)
    cv2.waitKey()

    return res