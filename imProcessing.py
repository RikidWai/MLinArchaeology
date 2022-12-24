import numpy as np
import cv2
import matplotlib.pyplot as plt
import imUtils
import configure as cfg
import colour
from colour_checker_detection.detection.segmentation import as_8_bit_BGR_image
import sys

folder = 'test_images/'
img = imUtils.imread(folder + '1.cr3')
img2 = img.copy()
img = imUtils.whiteBalance(img)
detector = cv2.mcc.CCheckerDetector_create()

if imUtils.detect24Checker(img.copy(), detector):
    # Color Correction
    checker = detector.getBestColorChecker()
    print('coor', checker.getBox())
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

    model.setWeightCoeff(1)

    model.run()

    ccm = model.getCCM()

    loss = model.getLoss()
    dst_rgbl = model.get_dst_rgbl()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float64)
    img_rgb = img_rgb/255
    # calibratedImage = model1.infer(img_)
    calibrated = model.infer(img_rgb)
    calibrated = imUtils.toOpenCVU8(calibrated.copy())

    # Detect black color
    # Crop the sherd
    patchPos = imUtils.getCardsPos(img.copy())

    filled, cnts = imUtils.masking(calibrated.copy())
    # edged = imUtils.getEdgedImg(img.copy())
    # imUtils.imshow(edged)
    # kernel = np.ones((5, 5), np.uint8)
    # dilation = cv2.dilate(edged, kernel, iterations=1)
    # imUtils.imshow(dilation)
    # (cnts, _) = cv2.findContours(dilation.copy(),
    #                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 5)
    print('pos', patchPos)
    for pos in patchPos.values():
        print('this is pos', pos)
        (x, y, w, h) = pos
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 5)
    imUtils.imshow(img2)
    cnts = list(filter(lambda cnt: imUtils.isSherd(
        cnt, patchPos, img.copy()), cnts))
    try:
        max_cnt = max(cnts, key=cv2.contourArea)
    except:
        print("Cnt contains no value")
        sys.exit(1)
    x, y, w, h = cv2.boundingRect(max_cnt)
    img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
              x-imUtils.MARGIN:x+w+imUtils.MARGIN]

else:
    patchPos, EXTRACTED_RGB, REF_RGB = imUtils.get4PatchInfo(img.copy())
    # patchPos = imUtils.get4ColourPatchPos(img.copy())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
    # currently the bestprint("corrected Vandermonde:")
    calibrated = colour.colour_correction(
        img, EXTRACTED_RGB, REF_RGB, 'Vandermonde')
    img = imUtils.toOpenCVU8(calibrated.copy())

    # FIXME: How to convert back to proper datatype for openCV to process?
    # img = cv2.cvtColor(corrected.astype(np.float32), cv2.COLOR_RGB2BGR)
    # img = imUtils.toOpenCVU8(corrected, True)
    # img = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    # imUtils.imshow(img)
    # img *= 255  # or any coefficient
    # img = img.astype(np.uint8)
    # img = img.astype(np.uint8)
    # corrected *= 255
    # corrected = corrected.astype(np.uint8)
    # img = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
    # imUtils.imshow(img)
    # img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # # img = img.astype(np.uint8)*255
    # imUtils.imshow(img)
    # Crop the sherd
    # temp = imUtils.correctedToU8(img.copy())
    # print('this is temp')
    # imUtils.imshow(temp)
    # gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    # _, thresh = cv2.threshold(
    #     gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # imUtils.imshow(thresh)

    # edged = imUtils.getEdgedImg(img.copy())
    _, cnts = imUtils.masking(img.copy())

    cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))
    #checking if max() arg is empty also filter out the unqualified images (e.g. ones with no colorChecker)
    try:
        max_cnt = max(cnts, key=cv2.contourArea)
    except:
        print("Cnt contains no value")
        sys.exit(1)
    x, y, w, h = cv2.boundingRect(max_cnt)
    img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
              x-imUtils.MARGIN:x+w+imUtils.MARGIN]

# Find Mask
# FIXME: better way of finding masks?
# edged = imUtils.getEdgedImg(img.copy())
# # threshold
# _, thresh = cv2.threshold(edged, 128, 255, cv2.THRESH_BINARY)
# # Find Mask
# FIXME: better way of finding masks?
# edged = imUtils.getEdgedImg(img.copy())


filled, max_cnt = imUtils.masking(img.copy(), 'biggest')
x, y, w, h = cv2.boundingRect(max_cnt)

mask = filled[imUtils.MARGIN:y+h-imUtils.MARGIN,
              imUtils.MARGIN:x+w-imUtils.MARGIN]
imUtils.imshow(mask, 'mask')
img = img[imUtils.MARGIN:y+h-imUtils.MARGIN,
          imUtils.MARGIN:x+w-imUtils.MARGIN]

sub_imgs = []

h, w = img.shape[0], img.shape[1]

for i in range(500):
    # if the object is too small
    if cfg.MAX_WIDTH > w or cfg.MAX_HEIGHT > h:
        break
    # if enough sub images are extracted
    if len(sub_imgs) == cfg.SAMPLE_NUM:
        break
    x1 = np.random.randint(0, w - cfg.MAX_WIDTH)
    y1 = np.random.randint(0, h - cfg.MAX_HEIGHT)

    # Extract the region only if it is within the mask
    if np.all(mask[y1: y1 + cfg.MAX_HEIGHT, x1: x1 + cfg.MAX_WIDTH]):
        sub_img = img[y1: y1 + cfg.MAX_HEIGHT,
                      x1: x1 + cfg.MAX_WIDTH, :]
        sub_imgs.append(sub_img)
    if i == 500 and len(sub_imgs) == 0:
        print('nth found!')

# Save the cropped regions
for i, sub_img in enumerate(sub_imgs):
    imUtils.imshow(sub_img)
# cv2.imwrite(f'{i + 1}.jpg', sub_img)
