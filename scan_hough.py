#!/usr/bin/env python

#
# Copyright (C) 2016 Fabrizio Colonna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language
# governing permissions and limitations under the License.
#

#
# Detects the document using a Hough transform and cluster detection 
#

# Import the necessary packages
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
from pprint import pprint

from utils import *
import numpy as np
import argparse
import math
import cv2

WORKING_HEIGHT = 700
BLUR_KERNEL = WORKING_HEIGHT / 100 + WORKING_HEIGHT / 100 % 2 - 1
CANNY_MINVAL = 80
CANNY_MAXVAL = 150
CANNY_SOBEL_SIZE = 5
MAX_ROTATION = np.pi / 4

def pre_process(image):
    image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, BLUR_KERNEL)
    
    return image

def get_line_mq(rho, theta):
    m = - math.cos(theta) / math.sin(theta)
    q = rho / math.sin(theta)
    return m, q


def rt_to_xy(rho, theta):
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

print(image)

thresh_image = image.copy()
ret, thresh_image = cv2.threshold(thresh_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
display_thresh = cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2RGB)
show("Threshold", thresh_image)

# Edges
image = cv2.Canny(image, CANNY_MINVAL, CANNY_MAXVAL, apertureSize=3)
show("Edges", image)
display_edges = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


#
# Lines detection
#

# Vertical
lines_v = cv2.HoughLines(image, 1, theta = np.pi / 180,
    threshold = 70,
    min_theta = -MAX_ROTATION,
    max_theta = +MAX_ROTATION
)
lines_v = np.array([[x[0][0], x[0][1]] for x in lines_v])
af_v = AffinityPropagation(preference=-50).fit(lines_v)

# Vertical
lines_h = cv2.HoughLines(image, 1, theta = np.pi / 180,
    threshold = 70,
    min_theta = np.pi / 2 - MAX_ROTATION,
    max_theta = np.pi / 2 + MAX_ROTATION
)
lines_h = np.array([[x[0][0], x[0][1]] for x in lines_h])
af_h = AffinityPropagation(preference=-50).fit(lines_h)

# All lines
lines = af_v.cluster_centers_.tolist() + af_h.cluster_centers_.tolist()

display_image = orig_image.copy()
for rho, theta in lines:
    x1, y1, x2, y2 = rt_to_xy(rho, theta)
    display_image = cv2.line(display_image, (x1,y1), (x2,y2), (0, 255, 255), 2)
    display_thresh = cv2.line(display_thresh, (x1,y1), (x2,y2), (255, 0, 0), 1) 
    display_edges = cv2.line(display_edges, (x1,y1), (x2,y2), (255, 0, 0), 1)
show("Image", display_image)
show("Image", display_thresh)
show("Image", display_edges)

for rho,theta in lines:
    #TODO: handle the case
    if (theta == 0):
        continue

    m, q = get_line_mq(rho,theta)
    for x in range(0, int(ratio * WORKING_HEIGHT)):
        y = round(m * x + q)
        print((x,y))


