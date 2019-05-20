import cv2
import numpy as np
import shutil
import os
from os.path import isfile, join

rootdir = './data/'
datadir = 'result1.mp4'

def main():
    dir = os.path.join(rootdir,datadir)

    if (os.path.isfile(dir)):
        img = cv2.imread(dir,cv2.IMREAD_COLOR)
    else:
        return -1

    img_original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    for i in xrange(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    maxline = lines[0]
    minline = lines[0]
    for line in lines:
        if maxline

    res = np.vstack((img_original, img))
    cv2.imshow('img', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()