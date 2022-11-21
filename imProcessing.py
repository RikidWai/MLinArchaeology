import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt
import math
import imUtils
import configure as cfg

folder = 'test_images/'
img = imUtils.imread(folder + '1.cr3')

# Color Correction
img = imUtils.whiteBalance(img)
img = imUtils.contrastStretching(img)


patchPos = imUtils.get4ColourPatchPos(img.copy())

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

while len(sub_imgs) < cfg.SAMPLE_NUM:
    if cfg.MAX_WIDTH > w or cfg.MAX_HEIGHT > h:
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
