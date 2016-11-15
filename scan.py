#!/usr/bin/env python

# Import the necessary packages
from skimage.filter import threshold_adaptive
from transform import four_point_transform
import imutils
import numpy as np
import argparse
import cv2

def pre_process(img):
	img   = img.copy()
	clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imshow("Grayscale", img)

	#img = clahe.apply(img)
	#cv2.imshow("Equalized", img)

	img = cv2.GaussianBlur(img, (3, 3), 0)
	img = cv2.Canny(img, 100, 250, 5)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3, 3))
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
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height = 500)
orig  = image.copy()

image = pre_process(image)

# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
im2, cnts, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE )
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

# loop over the contours
for c in cnts:
	# approximate the contour
	approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)

	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) < 4 or cv2.contourArea(approx) < 500: continue

	print("Area: %s" % cv2.contourArea(approx))
	print("Points: %s" % len(approx))
	print

	# show the contour (outline) of the piece of paper
	with_contours = orig.copy()
	cv2.drawContours(with_contours, [approx], -1, (0, 0, 255), 1)
	cv2.drawContours(with_contours, approx, -1, (255, 0, 0), 3)
	cv2.imshow("With contours", with_contours)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	with_contours = orig.copy()
	ch = cv2.convexHull(approx)
	cv2.drawContours(with_contours, [ch], -1, (0, 0, 255), 1)
	cv2.drawContours(with_contours, ch, -1, (255, 0, 0), 3)
	cv2.imshow("Convext Hull", with_contours)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

	if len(approx) == 4:
		screenCnt = approx
		break

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
