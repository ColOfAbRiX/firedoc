#!/usr/bin/env python

# Import the necessary packages
import numpy as np
import argparse
import cv2
import math
from utils import *

WORKING_HEIGHT = 700
BLUR_KERNEL = WORKING_HEIGHT / 100 + WORKING_HEIGHT / 100 % 2 - 1
#CLAHE_CLIP_LIMIT = 2.0
#CLAHE_GRID_SIZE = 8
CANNY_MINVAL = 80
CANNY_MAXVAL = 150
CANNY_SOBEL_SIZE = 5
MAX_ROTATION = np.pi / 6

def pre_process(image):
    image = image.copy()

    # Grescale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE
    #clahe = cv2.createCLAHE(clipLimit = CLAHE_CLIP_LIMIT, tileGridSize = (CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
    #image = clahe.apply(image)

    # Blur
    image = cv2.medianBlur(image, BLUR_KERNEL)

    return image

def rt2xy(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return (x1, y1, x2, y2)

def computeIntersect(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    x = -1
    y = -1

    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

    if d != 0.0:
        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d

    return (x, y)

#
# Loading
#

# Get arguments from command line
ap = argparse.ArgumentParser(description = 'Process some integers.')
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# Loading
image = cv2.imread(args["image"])
image = resize(image, height = WORKING_HEIGHT)
ratio = image.shape[0] / float(WORKING_HEIGHT)
orig_image  = image.copy()

# Pre-processing
image = pre_process(image)
show("Image", image)

# Edges
image = cv2.Canny(image, CANNY_MINVAL, CANNY_MAXVAL, apertureSize=3)
show("Edges", image)

lines_v = cv2.HoughLines(
    image = image,
    rho = 1,
    theta = np.pi / 180,
    threshold = 75,
    min_theta = -MAX_ROTATION,
    max_theta = +MAX_ROTATION
)

lines_h = cv2.HoughLines(
    image = image,
    rho = 1,
    theta = np.pi / 180,
    threshold = 75,
    min_theta = np.pi / 2 - MAX_ROTATION,
    max_theta = np.pi / 2 + MAX_ROTATION
)

for l in lines_v:
    x1, y1, x2, y2 = rt2xy(l[0][0], l[0][1])
    orig_image = cv2.line(orig_image, (x1,y1), (x2,y2), (0, 255, 0), 1)

for l in lines_h:
    x1, y1, x2, y2 = rt2xy(l[0][0], l[0][1])
    orig_image = cv2.line(orig_image, (x1,y1), (x2,y2), (0, 0, 255), 1)

show("Image", orig_image)

#
#for line in lines:
#    pass
#
## Find intersections
#corners = []
#for l1 in lines:
#    for l2 in lines:
#        pt = computeIntersect(l1, l2)
#        if pt[0] >= 0 and pt[1] >= 0:
#            corners.append(pt)
