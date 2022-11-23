import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt
import math

# define range of colors in HSV
lower_blue = np.array([120, 50, 20])
upper_blue = np.array([158, 255, 255])
lower_green = np.array([40, 52, 70])
upper_green = np.array([102, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([161, 100, 100])
upper_red2 = np.array([179, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 60])
lower_white = np.array([0, 0, 180])  # TOFIX
upper_white = np.array([0, 0, 255])  # TOFIX

# Create an array specify lower and upper range of colours
COLOUR_RANGE = [[[lower_blue, upper_blue]],
                [[lower_green, upper_green]],
                [[lower_yellow, upper_yellow]],
                [[lower_red, upper_red], [lower_red2, upper_red2]],
                [[lower_black, upper_black]]]

MARGIN = 2  # Margin for cropping


def imread(path, percent=50):
    _, extension = os.path.splitext(path)
    try:
        if 'cr' in extension or 'CR' in extension:
            raw = rawpy.imread(path)  # access to the RAW image
            rgb = raw.postprocess()  # a numpy RGB array
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # the OpenCV image

            scale_percent = percent  # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            img = cv2.resize(img, dim)
        else:
            img = cv2.imread(path)
        return img
    except Exception as e:
        print(e)


def imshow(img):
    if img is None:
        raise Exception('No Image')
    if img.shape[1] >= 1000 and img.shape[1] >= img.shape[0]:
        width = 1000
        height = int(img.shape[0] * 1000 / img.shape[1])
        dim = (width, height)
        img = cv2.resize(img, dim)
    elif img.shape[0] >= 1000 and img.shape[0] >= img.shape[1]:
        width = int(img.shape[1] * 1000 / img.shape[0])
        height = 1000
        dim = (width, height)
        img = cv2.resize(img, dim)
    try:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    except Exception as e:
        print(e)


def getEdgedImg(img):
    blur = cv2.medianBlur(img, 3)
    med_val = np.median(img)
    lower = int(max(0, 0.5*med_val))
    upper = int(min(255, 1.3*med_val))
    edged = cv2.Canny(blur, lower, upper)
    return edged

# Color Correction


def closest_color_to_white(COLORS):
    r, g, b = 255, 255, 255
    color_diffs = []
    for color in COLORS:
        cr, cg, cb = color
        color_diff = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]


def whiteBalance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - \
        ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - \
        ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def contrastStretching(img):
    norm_img = cv2.normalize(img, None, alpha=0, beta=1,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img = (255*norm_img).astype(np.uint8)
    return norm_img

# Cropping


def get4ColourPatchPos(img):

    img = whiteBalance(img)
    img = contrastStretching(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV

    patchPos = []

    for i in range(len(COLOUR_RANGE)):
        # Threshold the HSV image to get only certain colors
        if len(COLOUR_RANGE[i]) == 1:
            mask = cv2.inRange(
                img_hsv, COLOUR_RANGE[i][0][0], COLOUR_RANGE[i][0][1])
        else:  # Red color
            mask_1 = cv2.inRange(
                img_hsv, COLOUR_RANGE[i][0][0], COLOUR_RANGE[i][0][1])
            mask_2 = cv2.inRange(
                img_hsv, COLOUR_RANGE[i][1][0], COLOUR_RANGE[i][1][1])
            mask = cv2.bitwise_or(mask_1, mask_2)
        # Find contours and filter using threshold area
        colour_cnts, _ = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangle only
        colour_cnts = list(filter(lambda x: len(cv2.approxPolyDP(
            x, 0.01*cv2.arcLength(x, True), True)) == 4, colour_cnts))

        # Get the largest area
        if len(colour_cnts) > 0:
            coloured_cnt = max(colour_cnts, key=cv2.contourArea)

        if cv2.contourArea(coloured_cnt) > 400:

            # offsets - with this you get 'mask'
            x, y, w, h = cv2.boundingRect(coloured_cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            patchPos.append((x, y, w, h))

    return patchPos


def isSherd(cnt, patchPos):
    x, y, w, h = cv2.boundingRect(cnt)
    global img

    for pos in patchPos:
        if w > 100 and h > 100:  # Filter those small edges detected
            # Axis-Aligned Bounding Box
            # Test if two bound box not intersect
            # Return True is sherd
            if (x + w) < pos[0] or x > (pos[0] + pos[2]) or y > (pos[1] + pos[3]) or (y + h) < pos[1]:
                return True
    return False
