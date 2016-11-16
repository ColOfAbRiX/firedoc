#!/usr/bin/env python

# Import the necessary packages
from skimage.filter import threshold_adaptive
from transform import four_point_transform
import imutils
import numpy as np
import argparse
import cv2

# Parameters
CLAHE_CLIP_LIMIT = 1.5
CLAHE_GRID_SIZE = 8
BLUR_KERNEL = 7
CANNY_MINVAL = 100
CANNY_MAXVAL = 150
CANNY_SOBEL_SIZE = 3
DILATE_KERNEL = 20
DILATE_ITERATIONS = 1
APPROX_PRECISION = 0.02
COVERING_AREA = 0.25

def pre_process(img):
	img   = img.copy()
	clahe = cv2.createCLAHE(clipLimit = CLAHE_CLIP_LIMIT, tileGridSize = (CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = clahe.apply(img)
	cv2.imshow("Equalized", img)

	img = cv2.medianBlur(img, BLUR_KERNEL)
	img = cv2.Canny(img, CANNY_MINVAL, CANNY_MAXVAL, CANNY_SOBEL_SIZE)
	img = cv2.dilate(img, (DILATE_KERNEL, DILATE_KERNEL), iterations = DILATE_ITERATIONS)
	cv2.imshow("Closed", img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return img

# Get arguments from command line
ap = argparse.ArgumentParser(description = 'Process some integers.')
ap.add_argument("-i", "--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# Loading
image = cv2.imread(args["image"])
image = imutils.resize(image, height = 500)

width, height = image.shape[:2]
image_area = width * height

orig  = image.copy()
image = pre_process(image)

def contour_metric(contour):
	return cv2.contourArea(cv2.convexHull(contour))

# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
im2, contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
contours = sorted(contours, key = contour_metric, reverse = True)

# loop over the contours
big_shapes = []
for c in contours:
	# approximate the contour
	approx = cv2.approxPolyDP(c, APPROX_PRECISION * cv2.arcLength(c, True), True)
	hull = cv2.convexHull(approx)
	hull_area = cv2.contourArea(hull)

	if len(hull) > 4 and hull_area >= image_area * COVERING_AREA:
		big_shapes.append(hull)


# Order by number of points, prefer less points
contours = sorted(big_shapes, key = len)

if len(contours) == 0:
	print "No paper-shape found. Exit."
	exit()

paper = contours[0]
hull = cv2.convexHull(paper)

# show the contour (outline) of the piece of paper
with_contours = orig.copy()
cv2.drawContours(with_contours, [paper], -1, (0, 0, 255), 2)
cv2.drawContours(with_contours, paper, -1, (255, 0, 0), 4)
cv2.drawContours(with_contours, [hull], -1, (0, 255, 0), 2)
cv2.drawContours(with_contours, hull, -1, (0, 255, 0), 4)
cv2.imshow("With contours", with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

# apply the four point transform to obtain a top-down
# view of the original image
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
 
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 251, offset = 10)
warped = warped.astype("uint8") * 255
 
# show the original and scanned images
print "STEP 3: Apply perspective transform"
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)
