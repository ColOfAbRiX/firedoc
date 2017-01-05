#!/usr/bin/env python

import cv2

def show(title, image, contour = [], colour = (0, 0, 255), thickness = 2):
    image = image.copy()

    if contour != []:
        cv2.drawContours(image, [contour], -1, colour, thickness)
    
    cv2.imshow(title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized
