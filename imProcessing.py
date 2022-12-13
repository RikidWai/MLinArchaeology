import numpy as np
import cv2
import rawpy
import os
import matplotlib.pyplot as plt
import math
import imUtils
import configure as cfg
import colour
import skimage.transform

folder = 'test_images/'
img = imUtils.imread(folder + '1.cr3')
img2 = img.copy()
img = imUtils.whiteBalance(img)

detector = cv2.mcc.CCheckerDetector_create()

if imUtils.detect24Checker(img.copy(), detector):
    # Color Correction
    checker = detector.getBestColorChecker()
    cdraw = cv2.mcc.CCheckerDraw_create(checker)
    img_draw = cdraw.draw(img.copy())
    imUtils.imshow(img_draw)
    chartsRGB = checker.getChartsRGB()

    src = chartsRGB[:, 1].copy().reshape(24, 1, 3)
    src /= 255.0
    print(src.shape)

    # model1 = cv2.ccm_ColorCorrectionModel(src, cv2.mcc.MCC24)
    model = cv2.ccm_ColorCorrectionModel(
        src, imUtils.chartsRGB_np, cv2.ccm.COLOR_SPACE_sRGB)
    # model1.run()

    model.setWeightCoeff(1)

    model.run()

    ccm = model.getCCM()

    loss = model.getLoss()
    print("loss model 2 ", loss)
    dst_rgbl = model.get_dst_rgbl()
    print("mask", model.get_src_rgbl())

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float64)
    img_rgb = img_rgb/255
    # calibratedImage = model1.infer(img_)
    calibratedImg = model.infer(img_rgb)

    # out_ = calibratedImage * 255
    out = calibratedImg * 255

    # out_[out_ < 0] = 0
    out[out < 0] = 0

    # out_[out_ > 255] = 255
    out[out > 255] = 255

    # out_ = out_.astype(np.uint8)
    out = out.astype(np.uint8)

    # out_img = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    imUtils.imshow(img)

    # Detect black color
    # Crop the sherd
    patchPos = imUtils.getCardsPos(img)
    edged = imUtils.getEdgedImg(img.copy())
    imUtils.imshow(edged)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edged, kernel, iterations=1)
    imUtils.imshow(dilation)
    (cnts, _) = cv2.findContours(dilation.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 2)
    imUtils.imshow(img2)
    # cnts = filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts)
    # max_cnt = max(cnts, key=cv2.contourArea)
    # x, y, w, h = cv2.boundingRect(max_cnt)
    # img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
    #           x-imUtils.MARGIN:x+w+imUtils.MARGIN]

else:
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

    # FIXME: How to convert back to proper datatype for openCV to process?
    corrected = cv2.normalize(src=corrected, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    imUtils.imshow(img)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # img = img.astype(np.uint8)*255
    imUtils.imshow(img)
    # Crop the sherd
    edged = imUtils.getEdgedImg(img.copy())
    (cnts, _) = cv2.findContours(edged.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts)

    max_cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
              x-imUtils.MARGIN:x+w+imUtils.MARGIN]

# # Find Mask
# FIXME: better way of finding masks?
# edged = imUtils.getEdgedImg(img.copy())

# # threshold
# thresh = cv2.threshold(edged, 128, 255, cv2.THRESH_BINARY)[1]

# # apply close morphology
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# # get bounding box coordinates from the one filled external contour
# filled = np.zeros_like(thresh)
# (cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# max_cnt = max(cnts, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(max_cnt)
# cv2.drawContours(filled, [max_cnt], 0, 255, -1)

# # crop filled contour image

# mask = filled[imUtils.MARGIN:y+h-imUtils.MARGIN,
#               imUtils.MARGIN:x+w-imUtils.MARGIN]
# img = img[imUtils.MARGIN:y+h-imUtils.MARGIN,
#           imUtils.MARGIN:x+w-imUtils.MARGIN]

# sub_imgs = []

# h, w = img.shape[0], img.shape[1]

# for i in range(500):
#     if cfg.MAX_WIDTH > w or cfg.MAX_HEIGHT > h:
#         break
#     if len(sub_imgs) == cfg.SAMPLE_NUM:
#         break
#     x1 = np.random.randint(0, w - cfg.MAX_WIDTH)
#     y1 = np.random.randint(0, h - cfg.MAX_HEIGHT)

#     # Extract the region only if it is within the mask
#     if np.all(mask[y1: y1 + cfg.MAX_HEIGHT, x1: x1 + cfg.MAX_WIDTH]):
#         sub_img = img[y1: y1 + cfg.MAX_HEIGHT,
#                       x1: x1 + cfg.MAX_WIDTH, :]
#         sub_imgs.append(sub_img)

# # Save the cropped regions
# for i, sub_img in enumerate(sub_imgs):
#     imUtils.imshow(sub_img)
# # cv2.imwrite(f'{i + 1}.jpg', sub_img)
