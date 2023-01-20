import numpy as np
import pandas as pd
import cv2
import rawpy
import os
import colour
import logging
from datetime import datetime

chartsRGB = [
    [[115, 83, 68]],
    [[196, 147, 127]],
    [[91, 122, 155]],
    [[94, 108, 66]],
    [[129, 128, 176]],
    [[98, 190, 168]],
    [[223, 124, 47]],
    [[72, 92, 174]],
    [[194, 82, 96]],
    [[93, 60, 103]],
    [[162, 190, 62]],
    [[229, 158, 41]],
    [[49, 66, 147]],
    [[77, 153, 71]],
    [[173, 57, 60]],
    [[241, 201, 25]],
    [[190, 85, 150]],
    [[0, 135, 166]],
    [[242, 243, 245]],
    [[203, 203, 204]],
    [[162, 163, 162]],
    [[120, 120, 120]],
    [[84, 84, 84]],
    [[50, 50, 52]],
]
chartsRGB_np = np.array(chartsRGB).astype(float) / 255.0

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
upper_black = np.array([179, 255, 70])
lower_white = np.array([0, 0, 180])  # TOFIX
upper_white = np.array([0, 0, 255])  # TOFIX

# Create an array specify lower and upper range of colours
COLOR_RANGE = {'blue': [lower_blue, upper_blue],
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
                  'black': [0, 0, 0],  # black
                  'white': [1, 1, 1], }  # white

# Read a raw image
def imread(path, scaling_factor=1):
    _, extension = os.path.splitext(path)
    try:
        if 'cr' in extension or 'CR' in extension:
            raw = rawpy.imread(path).postprocess()  # access to the RAW image to a numpy RGB array
            img = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)  # the OpenCV image
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
            # width = int(img.shape[1] * scale_percent / 100)
            # height = int(img.shape[0] * scale_percent / 100)
            # dim = (width, height)
            # img = cv2.resize(img, dim)
        else:
            img = cv2.imread(path)
        return img
    except Exception as e:
        print(e)

# Display an image
def imshow(img, title='img'):
    if img is None:
        raise Exception('No Image')
    size = 800
    if img.shape[1] >= size and img.shape[1] >= img.shape[0]:
        width = size
        height = int(img.shape[0] * size / img.shape[1])
        dim = (width, height)
        img = cv2.resize(img, dim)
    elif img.shape[0] >= size and img.shape[0] >= img.shape[1]:
        width = int(img.shape[1] * size / img.shape[0])
        height = size
        dim = (width, height)
        img = cv2.resize(img, dim)
    try:
        cv2.imshow(title, img)
        cv2.waitKey(0)
    except Exception as e:
        print(e)

# Draw contours of an image
def drawCnts(img, cnts):
    print("Number of Contours found = " + str(len(cnts)))
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)
    imshow(img)

def drawPatchPos(img, patchPos): 
    for color in patchPos:
        # rect_color = (0, 0, 255) if color == 'black' else (0, 255, 0)
        
        x, y, w, h = patchPos['black']
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 10)
    imshow(img)

# detect edges in an image
def getEdgedImg(img):
    img = toOpenCVU8(img)

    blur = cv2.medianBlur(img, 3)
    med_val = np.median(img)
    lower = int(max(0, 0.5*med_val))
    upper = int(min(255, 1.3*med_val))
    edged = cv2.Canny(blur, lower, upper)

    return edged

# validate contours that are big enough only
def validCnt(cnt):
    (width, height)= cv2.minAreaRect(cnt)[1]
    if width > 50 and height > 50 and cv2.contourArea(cnt) > 400: # Filter those small edges detected
        return True
    return False

# Convert data type of an image from float64 to uint8
def toOpenCVU8(img):
    out = img * 255
    out[out < 0] = 0
    out[out > 255] = 255
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out

# detect if 24-patch color card exists
def detect24Checker(img, detector, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    processParams = cv2.mcc.DetectorParameters_create()
    processParams.maxError = 0.05
    if not detector.process(closing, cv2.mcc.MCC24, 1, params=processParams):
        return False
    return True
# Color Correction

# a helper function to find the region contains color that is closest to white
def closest_color_to_white(COLORS):
    r, g, b = 255, 255, 255
    color_diffs = []
    for color in COLORS:
        cr, cg, cb = color
        color_diff = np.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        color_diffs.append((color_diff, color))
    return min(color_diffs)[1]

# apply white balance
def white_bal(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - \
        ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - \
        ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

# apply contrast stretching
def contrast_stretching(img):
    norm_img = cv2.normalize(img, None, alpha=0, beta=1,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img = (255*norm_img).astype(np.uint8)
    return norm_img

# Cropping

# apply masking 
def masking(img, kernel_size=6):
    # Resize to 50% of original size for better masking
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    blur = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4)
    # apply close morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # get bounding box coordinates from the one filled external contour
    filled = np.zeros_like(thresh)

    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cnt for cnt in cnts if validCnt(cnt)]
    for cnt in cnts:
        cv2.drawContours(filled, [cnt], 0, 255, -1)
    
    filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
    
    # Resize to original size 
    filled = cv2.resize(filled, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    cnts, _ = cv2.findContours(
        filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return filled, cnts

# Detect the black region to guess the positions of 24checker and scaling card in an image 
def getCardsBlackPos(img):

    patchPos = {}
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert BGR to HSV
    black_mask = cv2.inRange(
        img_hsv, COLOR_RANGE['black'][0], COLOR_RANGE['black'][1])
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = black_mask.copy()
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cnt for cnt in cnts if validCnt(cnt)]
    cnts_rect = list(filter(lambda x: len(cv2.approxPolyDP(
            x, 0.1*cv2.arcLength(x, True), True)) == 4, cnts))
    cnts_rect = sorted(cnts_rect, reverse=True, key=cv2.contourArea)
    num_cnts = len(cnts_rect)
    if num_cnts >= 2: 
        patchPos['black'] = cv2.boundingRect(cnts_rect[1]) # Second largest is the scale card 
        patchPos['black2'] = cv2.boundingRect(cnts_rect[0])
    # drawPatchPos(img.copy(), patchPos)

    # # Get the largest area
    # if len(cnts_rect) > 0:
    #     rect_cnt = max(cnts_rect, key=cv2.contourArea)
    # patchPos['black'] = cv2.boundingRect(rect_cnt)
    return patchPos

# Detect 4 patches in 4Checker for color calibration
def get4PatchInfo(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    patchPos = {}
    coloursRectList = {}
    EXTRACTED_RGB, REF_RGB = [], []
    meansOfWhiteRegions = []

    for color in COLOR_RANGE:
        # Threshold the HSV image to get only certain colors
        if color != 'red':
            mask = cv2.inRange(
                img_hsv, COLOR_RANGE[color][0], COLOR_RANGE[color][1])
        else:  # Red color
            mask_1 = cv2.inRange(
                img_hsv, COLOR_RANGE[color][0][0], COLOR_RANGE[color][0][1])
            mask_2 = cv2.inRange(
                img_hsv, COLOR_RANGE[color][1][0], COLOR_RANGE[color][1][1])
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
            x,y,w,h = cv2.boundingRect(coloured_cnt)
            patchPos[color] = (x,y,w,h)

            colorRect = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            colorRect_1D = np.vstack(colorRect)/255

            REF_RGB.append(
                ([REF_RGB_4Patch[color]]*colorRect_1D.shape[0]).copy())
            EXTRACTED_RGB.append(colorRect_1D.copy())

            coloursRectList[color] = (cv2.cvtColor(
                img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))

    # get the white patch by comparing the left,right,top,bottom patch from the red patch, see which one is the whitest
    if 'red' in patchPos:
        (x, y, w, h) = patchPos['red']

        # right side
        rightSide = img[y:y+h, x+w:x+w+w//2+w//6]
        right_mean = cv2.mean(rightSide)[0:3]
        meansOfWhiteRegions.append(right_mean)

        # left side
        leftSide = img[y:y+h, (x-w//2-w//6):x]
        left_mean = cv2.mean(leftSide)[0:3]
        meansOfWhiteRegions.append(left_mean)

        # top side
        topSide = img[(y-h//2-h//6):y, x:x+w]
        top_mean = cv2.mean(topSide)[0:3]
        meansOfWhiteRegions.append(top_mean)

        # bottom side
        bottomSide = img[y+h:y+h+h//2+h//6, x:x+w]
        bottom_mean = cv2.mean(bottomSide)[0:3]
        meansOfWhiteRegions.append(bottom_mean)

        whiteRegions = [rightSide, leftSide, topSide, bottomSide]
        whiteIndex = meansOfWhiteRegions.index(
            closest_color_to_white(meansOfWhiteRegions))

        colorRect = cv2.cvtColor(whiteRegions[whiteIndex], cv2.COLOR_BGR2RGB)
        colorRect_1D = np.vstack(colorRect)/255
        REF_RGB.append(([REF_RGB_4Patch['white']] *
                       colorRect_1D.shape[0]).copy())
        EXTRACTED_RGB.append(colorRect_1D.copy())
        coloursRectList['white'] = cv2.cvtColor(
            whiteRegions[whiteIndex], cv2.COLOR_BGR2RGB)

    EXTRACTED_RGB = np.array(np.vstack(EXTRACTED_RGB))
    REF_RGB = colour.cctf_decoding(np.array(np.vstack(REF_RGB)))

    return patchPos, EXTRACTED_RGB, REF_RGB

# Guess if a contour is a sherd
def isSherd(cnt, patchPos):
    x, y, w, h = cv2.boundingRect(cnt)

    for pos in patchPos.values():
        # Axis-Aligned Bounding Box
        # Test if two bound box not intersect
        # Return True is sherd
        if not ((x + w) < pos[0] or x > (pos[0] + pos[2]) or y > (pos[1] + pos[3]) or (y + h) < pos[1]):
            return False
    return True


# Rotates a numpy image by right angle
# Direction options counterclockwise 'ccw' and clockwise 'cw'
def rotate_right_angle(img, direction):
    if direction == 'ccw':
        out = cv2.transpose(img)
        out = cv2.flip(out, flipCode=0)
    elif direction == 'cw':
        out = cv2.transpose(img)
        out = cv2.flip(out, flipCode=1)
    else:
        print('Incorrect rotation direction!')
        return img
    return out


# Gives the list of contours from a binary image
# The binary image can be obtained from inRange, threshold, grayscale, etc
def get_contours(binary_img):
    contours_tuple = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_tuple[0] if len(contours_tuple) == 2 else contours_tuple[1] # versioning
    return contours


# Finds the centroid of a contour shape
def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy
    else:
        print('Division by zero, contour is bad. Returning (0, 0)')
        return 0, 0

# Takes in BGR numpy image
# dst_ppc is the destinated number of pixels per cm
# Outputs the scaled image and the scaling factor
def scale_img(img, dst_ppc=118, patchPos = None):
    # min_area = 400 # Min area for valid contour
    # kernel_size = 5 # Adjust for morphing
    # arclength_factor = 0.05 # Adjust for rectangle checking, 0.01 may not work that well

    # # Masking to retrieve black color parts
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # black_mask = cv2.inRange(img_hsv, lower_black, upper_black)

    # # Morphing to connect lines and reduce noise
    # # Close: Dilate --> Erode
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # mask = black_mask.copy()
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # # imshow(mask, 'sam')
    # # Try contour detection
    # try:
    #     cnts = get_contours(mask)
    # except Exception as e:
    #     print(f'Error detecting contours: {e}')
    #     return img, 1

    # # Filter contours by area
    # cnts_filtered = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]

    # # Takes only rectangular contours
    # cnts_rect = list(filter(lambda x: len(cv2.approxPolyDP(
    #         x, 0.1*cv2.arcLength(x, True), True)) == 4, cnts_filtered))

    # if len(cnts_rect) == 0:
    #     print('Unable to detect the required calibration cards, returning original image')
    #     return img, 1

    # # Area sort
    # cnts_sorted = sorted(cnts_rect, reverse=True, key=cv2.contourArea)

    # # Choose second biggest
    # idx = 1 if len(cnts_sorted) > 1 else 0

    # # Compute length of black square in 4_color, length of calibration card in 24_color
    # approx_chosen = cv2.approxPolyDP(cnts_sorted[idx], 0.01*cv2.arcLength(cnts_sorted[idx], True), True)

    # x, y, w, h = cv2.boundingRect(cnts_sorted[idx])
    # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 10)
    # imshow(img, 'sam black square')
    if 'black' not in patchPos:
        print('No black squares detected, returning original image')
        return img, 1
    
    (_, _, w, h) = patchPos['black']
    
    if w == 0 or h == 0:
        print('Width or height zero, error detecting bounding box of calibration card, returning original image')
        return img, 1
    # print(w,h)
    # Deals with the case of using 24 color card and 4 color card
    if w/h > 2 or h/w > 2: # 24c, 2nd biggest is calibration card, with length of card being 5cm
        ppc = max(w, h)/5
        # print(ppc)
    else: # 4c, 2nd biggest if two black squares found, else choose the only one. Each square is 1cm in length
        ppc = max(w, h)

    if ppc <= 0:
        print('Error obtaining detected ppc, returning original image')
        return img, 1


    scaling_factor = dst_ppc/ppc
   
    # Suspect when ppc is small e.g. 18, then may crash
    if scaling_factor >= 5 or (max(img.shape) > 3000 and scaling_factor >= 3):
        print(f'Scaling factor too large ({scaling_factor}), aborting scaling')
        return img, 1
    elif scaling_factor <= 0.35 or (min(img.shape) < 1000 and scaling_factor <= 0.5):
            print(f'Scaling factor too small ({scaling_factor}), aborting scaling')
            return img, 1
    try:
        img_scaled = cv2.resize(img.copy(), None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f'Error scaling image: {e}')
        return img, 1
    print(scaling_factor)
    return img_scaled, scaling_factor



# Logs error messages
def log_err(logger, err=None, msg=None):
    if err:
        logger.error(err)
    if msg:
        logger.info(msg)


# Creates logger object and log directory
def init_logger():
    log_dir = '../logs'
    log_path = log_dir + datetime.now().strftime('/img_process_log_%Y_%m_%d_%H_%M.log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, 
                            format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger = logging.getLogger('img_processing')
    return logger


# Appends to a list containing the data images that had errors in image processing
# file is the same argument passed to process()
def append_err_list(err_list, file):
    filepath = os.path.splitext(file)[0]
    filename = filepath.split('/')[-2] + "/" + filepath.split('/')[-1]
    err_list.append(filename)


# Saves error list as csv for inspection
def err_list_to_csv(err_list):
    err_dir = '../err_list'
    err_path = err_dir + datetime.now().strftime('/err_list_%Y_%m_%d_%H_%M.csv')
    if not os.path.exists(err_dir):
        os.makedirs(err_dir)
    err_df = pd.DataFrame(err_list)
    err_df.to_csv(err_path, index=True)


