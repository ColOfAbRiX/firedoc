#!/usr/bin/env python
# See: http://stackoverflow.com/questions/34981144/split-text-lines-in-scanned-document

# Import the necessary packages
import numpy as np
import argparse
import cv2
import math
from utils import *

# Parameters
WORKING_HEIGHT = 1000

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

# Grescale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold
#_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
threshold_image = image
show("Threshold", threshold_image)

# Contours detection
image = cv2.Canny(image, 80, 150)
image, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contour = np.array(
    [item for sublist in contours for item in sublist]
)
show("Cropping", orig_image, contour)

# Bounding Rectangle
rect = cv2.minAreaRect(contour)
rect_center = rect[0]
rect_width, rect_height = rect[1]
rect_angle = rect[2];
rect_angle_rad = rect_angle * math.pi / 180.0
show("Cropping", orig_image, np.int0(np.around(cv2.boxPoints(rect))))

if rect_angle < -45.0:
    rect_angle += 90.0

rotated_image = cv2.warpAffine(
    threshold_image,
    cv2.getRotationMatrix2D(rect_center, rect_angle, 1),
    (threshold_image.shape[1], threshold_image.shape[0]),
    borderMode=cv2.INTER_LINEAR,
    borderValue=cv2.BORDER_CONSTANT
)
show("Deskewed", rotated_image)

# Reducing to lines info
_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
image = cv2.reduce(rotated_image, 1, cv2.REDUCE_MIN)
_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show("Reduced", image)

# Find position and number of spaces
y = 0; space_width = 0; count = 0; spaces = []
for i, p in enumerate(image):
    p = p[0]
    if space_width == 0:
        if p > 170:
            space_width = 1
            count = 1
            y = i
    else:
        if p <= 85:
            space_width = 0
            spaces.append(y / count)
        else:
            space_width += 5
            count += 1
            y += i

# Print lines
image = rotated_image
for s in spaces:
    image = cv2.line(image, (0, s), (rotated_image.shape[1], s), (0, 255, 0))

show("Split", image)
