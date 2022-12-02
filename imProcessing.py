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
img = imUtils.whiteBalance(img)

detector = cv2.mcc.CCheckerDetector_create() 

if imUtils.detect24Checker(img.copy(), detector):
    # Color Correction
    checker = detector.getBestColorChecker()
    cdraw = cv2.mcc.CCheckerDraw_create(checker)
    img_draw = cdraw.draw(img.copy())
    imUtils.imshow(img_draw)
    chartsRGB = checker.getChartsRGB()

    src = chartsRGB[:,1].copy().reshape(24, 1, 3)
    src /= 255.0
    print(src.shape)

    # model1 = cv2.ccm_ColorCorrectionModel(src, cv2.mcc.MCC24)
    model2 = cv2.ccm_ColorCorrectionModel(src, imUtils.chartsRGB_np,cv2.ccm.COLOR_SPACE_sRGB)
    # model1.run()

    model2.setWeightCoeff(1)

    model2.run()

    ccm2 = model2.getCCM()

    loss2 = model2.getLoss()
    print("loss model 2 ", loss2)
    dst_rgbl = model2.get_dst_rgbl()
    print("mask", model2.get_src_rgbl())




    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ = img_.astype(np.float64)
    img_ = img_/255
    # calibratedImage = model1.infer(img_)
    calibratedImage2 = model2.infer(img_)

    # out_ = calibratedImage * 255
    out_2 = calibratedImage2 * 255

    # out_[out_ < 0] = 0
    out_2[out_2 < 0] = 0

    # out_[out_ > 255] = 255
    out_2[out_2 > 255] = 255

    # out_ = out_.astype(np.uint8)
    out_2 = out_2.astype(np.uint8)


    # out_img = cv2.cvtColor(out_, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(out_2, cv2.COLOR_RGB2BGR)

    imUtils.imshow(img)

    # Detect black color 
    # Crop the sherd
    patchPos = imUtils.getCardsPos(img)
    #FIXME: It is not good to detect objects using edges.... 
    # Shadow problem 
    # can't detect edges when file size too large... 
    edged = imUtils.getEdgedImg(img.copy())
    imUtils.imshow(edged)
    (cnts, _) = cv2.findContours(edged.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts)
    max_cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_cnt)
    img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
            x-imUtils.MARGIN:x+w+imUtils.MARGIN]

else: 
    patchPos, EXTRACTED_RGB, REF_RGB = imUtils.get4PatchInfo(img.copy())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255


    # currently the bestprint("corrected Vandermonde:")
    corrected = colour.colour_correction(
        img, EXTRACTED_RGB, REF_RGB, 'Vandermonde')
    colour.plotting.plot_image(
        corrected
    )

    #FIXME: How to convert back to proper datatype for openCV to process? 
    corrected *=255 
    corrected = corrected.astype(np.uint8)
    img = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    imUtils.imshow(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
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
# edged = imUtils.getEdgedImg(img.copy())

# # threshold
# thresh = cv2.threshold(edged, 128, 255, cv2.THRESH_BINARY)[1]
# _, thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# # apply close morphology
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# # get bounding box coordinates from the one filled external contour
# filled = np.zeros_like(thresh)
# (cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# imUtils.drawCnts(img.copy(), cnts)
# max_cnt = max(cnts, key=cv2.contourArea)
# x, y, w, h = cv2.boundingRect(max_cnt)
# cv2.drawContours(filled, [max_cnt], 0, 255, -1)

# # crop filled contour image

# mask = filled[imUtils.MARGIN:y+h-imUtils.MARGIN,
#               imUtils.MARGIN:x+w-imUtils.MARGIN]
# imUtils.imshow(mask)
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
