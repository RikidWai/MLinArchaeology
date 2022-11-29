import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt
import math
import colour

chartsRGB = [[[115,83,68]],[[196,147,127]],[[91,122,155]],[[94,108,66]],[[129,128,176]],[[98,190,168]],[[223,124,47]],[[72,92,174]],[[194,82,96]],[[93,60,103]],[[162,190,62]],[[229,158,41]],[[49,66,147]],[[77,153,71]],[[173,57,60]],[[241,201,25]],[[190,85,150]],[[0,135,166]],[[242,243,245]],[[203,203,204]],[[162,163,162]],[[120,120,120]],[[84,84,84]],[[50,50,52]]]
chartsRGB_np = np.array(chartsRGB).astype(float)/ 255.0

# define range of colors in HSV
lower_blue = np.array([120, 50, 20])
upper_blue = np.array([158, 255, 255])
lower_green = np.array([40, 52, 70])
upper_green = np.array([82, 255, 255])  # TOFIX
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([179, 255, 255])

lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 60])
lower_white = np.array([0, 0, 180])  # TOFIX
upper_white = np.array([0, 0, 255])  # TOFIX

# Create an array specify lower and upper range of colours
COLOUR_RANGE = {'blue': [lower_blue, upper_blue],
                'green': [lower_green, upper_green],
                'yellow': [lower_yellow, upper_yellow],
                'red': [[lower_red, upper_red], [lower_red2, upper_red2]],
                'black': [lower_black, upper_black]}

# reference colours in rgb, in values between 0 and 1
b_ref = np.array([26, 0, 165])/255
g_ref = np.array([30, 187, 22])/255
y_ref = np.array([252, 222, 10])/255
r_ref = np.array([240, 0, 22])/255
REF_RGB_4Patch = {'blue': b_ref,
                  'green': g_ref,
                  'yellow': y_ref,
                  'red': r_ref,  
                  'black': [0, 0, 0], # black
                  'white': [1, 1, 1], }  # white

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
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(img, kernel)
    blur = cv2.medianBlur(eroded, 3)
    med_val = np.median(eroded)
    lower = int(max(0, 0.5*med_val))
    upper = int(min(255, 1.3*med_val))
    edged = cv2.Canny(blur, lower, upper)
    return edged

# TODO
def detect24Checker(img, detector): 
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if not detector.process(closing ,cv2.mcc.MCC24,1) :
        print("24Chart not detected. Assume 4Chart is used\n")
        return False
    edged = getEdgedImg(img.copy())
    ## 3. Do morph-close-op and Threshold

    
    print("24Chart detected.\n")
    return True
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

def getCardsPos(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    img2 = img.copy()
    mask = cv2.inRange(img_hsv, COLOUR_RANGE['black'][0], COLOUR_RANGE['black'][1])
    patchPos = {}
    colour_cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    colour_cnts = sorted(colour_cnts, reverse=True, key=cv2.contourArea)
    for i in range(2):
        if cv2.contourArea(colour_cnts[i]) > 400: 

            x, y, w, h = cv2.boundingRect(colour_cnts[i])
            patchPos[i] = (x, y, w, h)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    imshow(img2)
    return patchPos

        
def get4PatchInfo(img):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    img2 = img.copy()
    patchPos = {}
    coloursRectList = {}
    EXTRACTED_RGB = REF_RGB = []

    detected_areas_mean_around_red = []
    for color in COLOUR_RANGE:
        # Threshold the HSV image to get only certain colors
        if color != 'red':
            mask = cv2.inRange(
                img_hsv, COLOUR_RANGE[color][0], COLOUR_RANGE[color][1])
        else:  # Red color
            mask_1 = cv2.inRange(
                img_hsv, COLOUR_RANGE[color][0][0], COLOUR_RANGE[color][0][1])
            mask_2 = cv2.inRange(
                img_hsv, COLOUR_RANGE[color][1][0], COLOUR_RANGE[color][1][1])
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
        # print(color)
        if cv2.contourArea(coloured_cnt) > 400:

            # offsets - with this you get 'mask'
            x, y, w, h = cv2.boundingRect(coloured_cnt)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
            patchPos[color] = (x, y, w, h)

            coloursRect = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            coloursRect_1D = np.vstack(coloursRect)/255
            print(coloursRect_1D.shape)
            print(type(coloursRect_1D))

            REF_RGB.append([REF_RGB_4Patch[color]]*coloursRect_1D.shape[0])
            EXTRACTED_RGB.append(coloursRect_1D.copy())
            a = coloursRect_1D.copy()
            coloursRectList[color] = (cv2.cvtColor(
                img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
            # colour_b = np.vstack(coloursRectList['blue'])/255
            # EXTRACTED_RGB2 = np.array(np.vstack(EXTRACTED_RGB[0]))
            # print(color, 
            #     EXTRACTED_RGB2,  
            #     np.array(np.vstack(np.vstack(coloursRectList['blue'].copy())/255)), 
            #     np.array(np.vstack((colour_b))))


    # get the white patch by comparing the left,right,top,bottom patch from the red patch, see which one is the whitest
    if 'red' in patchPos: 
        (x, y, w, h) = patchPos['red']
        print("the white colour")
        print("x", x)
        print("w:", w)
        # right side
        rightSide = img[y:y+h, x+w:x+w+w//2+w//6]
        cv2.rectangle(img2, (x+w, y), (x+w+w//2 +
                        w//6, y+h), (0, 255, 0), 2)
        right_mean = cv2.mean(rightSide)[0:3]
        print("mean:", right_mean)
        detected_areas_mean_around_red.append(right_mean)
        # left side
        leftSide = img[y:y+h, (x-w//2-w//6):x]

        cv2.rectangle(img2, (x-w//2-w//6, y), (x, y+h), (0, 255, 0), 2)
        left_mean = cv2.mean(leftSide)[0:3]
        print("mean:", left_mean)
        detected_areas_mean_around_red.append(left_mean)
        # top side
        topSide = img[(y-h//2-h//6):y, x:x+w]
        cv2.rectangle(img2, (x, (y-h//2-h//6)),
                        (x+w, y), (0, 255, 0), 2)
        top_mean = cv2.mean(topSide)[0:3]
        print("mean:", top_mean)
        detected_areas_mean_around_red.append(top_mean)

        # bottom side
        bottomSide = img[y+h:y+h+h//2+h//6, x:x+w]
        cv2.rectangle(img2, (x, (y+h)),
                        (x+w, y+h+h//2+h//6), (0, 255, 0), 2)
        bottom_mean = cv2.mean(bottomSide)[0:3]
        print("mean:", bottom_mean)
        detected_areas_mean_around_red.append(bottom_mean)

        all_sides_colours = [rightSide, leftSide, topSide, bottomSide]
        index_foundClosest_toWhite_colour = detected_areas_mean_around_red.index(
            closest_color_to_white(detected_areas_mean_around_red))
        print("index of the patch closest to white:",
                index_foundClosest_toWhite_colour)

        coloursRect = cv2.cvtColor(
            all_sides_colours[index_foundClosest_toWhite_colour], cv2.COLOR_BGR2RGB)
        coloursRect_1D = np.vstack(coloursRect)/255

        REF_RGB.append([REF_RGB_4Patch['white']]
                        * coloursRect_1D.shape[0])
        EXTRACTED_RGB.append(coloursRect_1D.copy())
        coloursRectList['white'] = cv2.cvtColor(
            all_sides_colours[index_foundClosest_toWhite_colour], cv2.COLOR_BGR2RGB)

    # EXTRACTED_RGB2 = np.array(np.vstack(tuple(EXTRACTED_RGB)))
    EXTRACTED_RGB2 = np.array(np.vstack((
        np.vstack(coloursRectList['blue'])/255,
        np.vstack(coloursRectList['green'])/255,
        np.vstack(coloursRectList['yellow'])/255,
        np.vstack(coloursRectList['red'])/255,
        np.vstack(coloursRectList['black'])/255,
        np.vstack(coloursRectList['white'])/255)))
    REF_RGB2 = colour.cctf_decoding(np.array(np.vstack(tuple(REF_RGB))))
    imshow(img2)

    #organise the image sampled colour patches into four arrays (R, Y, G, B, White), in values between 0 and 1
    colour_b = np.vstack(coloursRectList['blue'])/255
    colour_g = np.vstack(coloursRectList['green'])/255
    colour_y = np.vstack(coloursRectList['yellow'])/255
    colour_r = np.vstack(coloursRectList['red'])/255
    colour_wh = np.vstack(coloursRectList['white'])/255
    colour_bl = np.vstack(coloursRectList['black'])/255
    REF_RGB = colour.cctf_decoding(
        np.array(np.vstack(([REF_RGB_4Patch['blue']]*colour_b.shape[0],
                            [REF_RGB_4Patch['green']]*colour_g.shape[0],
                            [REF_RGB_4Patch['yellow']]*colour_y.shape[0],
                            [REF_RGB_4Patch['red']]*colour_r.shape[0],                        
                            [REF_RGB_4Patch['black']]*colour_bl.shape[0], 
                            [REF_RGB_4Patch['white']]*colour_wh.shape[0],)))
    )

    return patchPos, EXTRACTED_RGB2, REF_RGB


def isSherd(cnt, patchPos):
    x, y, w, h = cv2.boundingRect(cnt)

    for pos in patchPos.values():
        if w > 100 and h > 100:  # Filter those small edges detected
            # Axis-Aligned Bounding Box
            # Test if two bound box not intersect
            # Return True is sherd
            if (x + w) < pos[0] or x > (pos[0] + pos[2]) or y > (pos[1] + pos[3]) or (y + h) < pos[1]:
                return True
    return False
