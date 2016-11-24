#!/usr/bin/env python

# Import the necessary packages
from utils import *
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

import numpy as np
import argparse
import math
import cv2

WORKING_HEIGHT = 700
BLUR_KERNEL = WORKING_HEIGHT / 100 + WORKING_HEIGHT / 100 % 2 - 1
CANNY_MINVAL = 80
CANNY_MAXVAL = 150
CANNY_SOBEL_SIZE = 5
MAX_ROTATION = np.pi / 6

def pre_process(image):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, BLUR_KERNEL)
    
    return image

def rt_to_xy(rho, theta):
    a = np.cos(theta); b = np.sin(theta)
    
    x0 = a * rho; y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    
    return (x1, y1, x2, y2)

def computeIntersect(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    x = -1; y = -1
    d = ((x1 - x2)*(y3 - y4)) - ((y1 - y2)*(x3 - x4))
    if d != 0.0:
        x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / d
        y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / d
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

#
# Pre-processing
#
image = pre_process(image)
show("Image", image)

# Edges
image = cv2.Canny(image, CANNY_MINVAL, CANNY_MAXVAL, apertureSize=3)
show("Edges", image)

#
# Lines detectoin
#

lines = []
# Vertical lines
lines += cv2.HoughLines(image, 1, theta = np.pi / 180,
    threshold = 70,
    min_theta = -MAX_ROTATION,
    max_theta = +MAX_ROTATION
).tolist()
# Horizontal lines
lines += cv2.HoughLines(image, 1, theta = np.pi / 180,
    threshold = 70,
    min_theta = np.pi / 2 - MAX_ROTATION,
    max_theta = np.pi / 2 + MAX_ROTATION
).tolist()

display_image = orig_image.copy()
for l in lines:
    x1, y1, x2, y2 = rt_to_xy(l[0][0], l[0][1])
    display_image = cv2.line(display_image, (x1,y1), (x2,y2), (0, 255, 0), 1)
show("Image", display_image)

#
# Clustering
#

lines = np.array([[x[0][0], x[0][1]] for x in lines])

bandwidth = estimate_bandwidth(lines, quantile=0.2, n_samples=500)
ms = MeanShift(
    bandwidth=bandwidth,
    bin_seeding=True
)
ms.fit(lines)
lines = ms.cluster_centers_

display_image = orig_image.copy()
for rho, theta in lines:
    print("Rho: %f, Theta: %f" % (rho, theta))
    x1, y1, x2, y2 = rt_to_xy(rho, theta)
    display_image = cv2.line(display_image, (x1,y1), (x2,y2), (0, 255, 255), 2)
show("Image", display_image)
