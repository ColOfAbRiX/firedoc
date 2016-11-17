#!/usr/bin/env python

# Import the necessary packages
from transform import four_point_transform
import imutils
import numpy as np
import argparse
import cv2

# Parameters
WORKING_HEIGHT = 1000
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 4
BLUR_KERNEL = 9
CANNY_MINVAL = 150
CANNY_MAXVAL = 180
CANNY_SOBEL_SIZE = 5
DILATE_MORPH = 3
APPROX_PRECISION = 0.02
COVERING_AREA = 0.25

def show(title, image, contour = []):
	image = image.copy()

	if len(contour) > 0:
		cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
	
	cv2.imshow(title, image)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

def pre_process(image):
	image = image.copy()

	# Grescale
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# CLAHE
	clahe = cv2.createCLAHE(clipLimit = CLAHE_CLIP_LIMIT, tileGridSize = (CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
	#mage = clahe.apply(image)

	# Blur
	image = cv2.medianBlur(image, BLUR_KERNEL)

 	# Threshold
 	#image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11, 2)
 	ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	return image

def find_edges(image):
	image = image.copy()

	# Edge detection
	image = cv2.Canny(image, CANNY_MINVAL, CANNY_MAXVAL, CANNY_SOBEL_SIZE)

	# Dilation
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_MORPH, DILATE_MORPH))
	image = cv2.dilate(image, kernel)

	return image

def find_contours(image):
	image = image.copy()

	im2, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
	contours = sorted(contours, key = lambda c: cv2.contourArea(cv2.convexHull(c)), reverse = True)
	
	return contours

def approx_hull(contours, image_area):
	big_shapes = []
	for c in contours:
		approx = cv2.approxPolyDP(c, APPROX_PRECISION * cv2.arcLength(c, True), True)
		hull = cv2.convexHull(approx)
		hull_area = cv2.contourArea(hull)

		if len(hull) >= 4 and hull_area >= image_area * COVERING_AREA:
			big_shapes.append(hull)

	if len(big_shapes) == 0:
		return []

	# Order by number of points, prefer less points
	return sorted(big_shapes, key = len)[0]

def approx_rect(contour):
	skewed_rect = []

	min_area_rect = np.int0(cv2.boxPoints(cv2.minAreaRect(contour)))
	
	for x in min_area_rect:
		dists = [(y ,cv2.norm(x, y[0])) for y in contour]
		dists = sorted(dists, key = lambda x: x[1])
		skewed_rect.append(dists[0][0])

	return min_area_rect
	return np.array(skewed_rect)

def deskew(image, rect):
	image = image.copy()

	image = four_point_transform(image, rect.reshape(4, 2) * ratio)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 	ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	return image


# Get arguments from command line
ap = argparse.ArgumentParser(description = 'Process some integers.')
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# Loading
image = cv2.imread(args["image"])
image = imutils.resize(image, height = WORKING_HEIGHT)
ratio = image.shape[0] / float(WORKING_HEIGHT)
orig  = image.copy()

#
# Pre-processing
#

image = pre_process(image)
show("Pre-processed", image)

#
# Contouring
#

# Find edges
image = find_edges(image)
show("Edges", image)

# Find contours
contours = find_contours(image)
image_contours = orig.copy()
if len(contours) == 0:
	print "No paper-shape found. Exit."
	exit()
show("Contour", orig, contours[0])

#
# Perspective Approximation and correction
#

# Find best approximated hull
image_area = image.shape[0] * image.shape[1]
contour = approx_hull(contours, image_area)
if len(contour) == 0:
	print "No paper-shape found. Exit."
	exit()
show("Approx Hull", orig, contour)

# Find best approximate skewed rectangle
rectangle = approx_rect(contour)
show("Skewed Rectangle", orig, rectangle)

# Correct perspective
image = deskew(orig, rectangle)
show("Deskewed Image", image)
