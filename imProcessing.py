import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt
import math
import imUtils
import configure as cfg
import colour

folder = 'test_images/'
img = imUtils.imread(folder + '1.cr3')
img2 = imUtils.imread(folder+'1.cr3', 100)
img = imUtils.whiteBalance(img)

# Color Correction
if imUtils.detect24Checker(img.copy()):
    print('24Checker is detected') 
else: 
    coloursRect = {}
    patchPos = []

    patchPos, EXTRACTED_RGB, REF_RGB = imUtils.get4PatchInfo(
        img.copy())
    # patchPos = imUtils.get4ColourPatchPos(img.copy())


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255


    # currently the bestprint("corrected Vandermonde:")
    corrected = colour.colour_correction(
        img, EXTRACTED_RGB, REF_RGB, 'Vandermonde')
    colour.plotting.plot_image(
        corrected
    )
    img = (cv2.cvtColor(corrected.astype(np.float32), cv2.COLOR_RGB2BGR)*255)
    img = img.astype(np.uint8)

    # Crop the sherd
    edged = imUtils.getEdgedImg(img.copy())

    (cnts, _) = cv2.findContours(edged.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts)

    max_cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
            x-imUtils.MARGIN:x+w+imUtils.MARGIN]

# Find Mask
edged = imUtils.getEdgedImg(img.copy())

# threshold
thresh = cv2.threshold(edged, 128, 255, cv2.THRESH_BINARY)[1]

# apply close morphology
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# get bounding box coordinates from the one filled external contour
filled = np.zeros_like(thresh)
(cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_cnt = max(cnts, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(max_cnt)
cv2.drawContours(filled, [max_cnt], 0, 255, -1)

# crop filled contour image

mask = filled[imUtils.MARGIN:y+h-imUtils.MARGIN,
              imUtils.MARGIN:x+w-imUtils.MARGIN]
img = img[imUtils.MARGIN:y+h-imUtils.MARGIN,
          imUtils.MARGIN:x+w-imUtils.MARGIN]

sub_imgs = []

h, w = img.shape[0], img.shape[1]

for i in range(500):
    if cfg.MAX_WIDTH > w or cfg.MAX_HEIGHT > h:
        break
    if len(sub_imgs) == cfg.SAMPLE_NUM:
        break
    x1 = np.random.randint(0, w - cfg.MAX_WIDTH)
    y1 = np.random.randint(0, h - cfg.MAX_HEIGHT)

    # Extract the region only if it is within the mask
    if np.all(mask[y1: y1 + cfg.MAX_HEIGHT, x1: x1 + cfg.MAX_WIDTH]):
        sub_img = img[y1: y1 + cfg.MAX_HEIGHT,
                      x1: x1 + cfg.MAX_WIDTH, :]
        sub_imgs.append(sub_img)

# Save the cropped regions
for i, sub_img in enumerate(sub_imgs):
    imUtils.imshow(sub_img)
# cv2.imwrite(f'{i + 1}.jpg', sub_img)
