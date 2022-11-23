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
coloursRect = []
patchPos = []

patchPos, coloursRect, EXTRACTED_RGB, REF_RGB = imUtils.get4PatchInfo(
    img.copy())
print(len(patchPos), len(coloursRect))
# patchPos = imUtils.get4ColourPatchPos(img.copy())

# organise the image sampled colour patches into four arrays (R, Y, G, B, White), in values between 0 and 1
colour_b = np.vstack(coloursRect[0])/255
colour_g = np.vstack(coloursRect[1])/255
colour_y = np.vstack(coloursRect[2])/255
colour_r = np.vstack(coloursRect[3])/255
colour_wh = np.vstack(coloursRect[4])/255
colour_bl = np.vstack(coloursRect[5])/255
REF_RGB = colour.cctf_decoding(
    np.array(np.vstack(([imUtils.REF_RGB_4Patch[0]]*colour_b.shape[0],
                        [imUtils.REF_RGB_4Patch[1]]*colour_g.shape[0],
                        [imUtils.REF_RGB_4Patch[2]]*colour_y.shape[0],
                        [imUtils.REF_RGB_4Patch[3]]*colour_r.shape[0],
                        [imUtils.REF_RGB_4Patch[4]]*colour_wh.shape[0],
                        [imUtils.REF_RGB_4Patch[5]]*colour_bl.shape[0])))
)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
EXTRACTED_RGB = np.array(
    np.vstack((colour_b, colour_g, colour_y, colour_r, colour_wh, colour_bl)))
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
