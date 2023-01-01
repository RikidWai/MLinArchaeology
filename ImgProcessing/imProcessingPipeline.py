import sys
sys.path.append('../')

import numpy as np
import cv2
import imUtils
import configure as cfg
import colour
import math
import os

DEFAULT_OUT_DIM = (1000, 500)

def improcessing(file, logger, err_list):
    out_w, out_h = DEFAULT_OUT_DIM
    img = imUtils.imread(file)

    img = imUtils.white_bal(img)
    detector = cv2.mcc.CCheckerDetector_create()
    
    # Scale image according to required ppc ratio
    try:
        img, scaling_factor = imUtils.scale_img(img, cfg.DST_PPC)
    except Exception as e:
        print(f'Error scaling image: {e}')
        imUtils.log_err(logger, err=e)
        imUtils.append_err_list(err_list,file)
        return

    if imUtils.detect24Checker(img.copy(), detector):
        # Color Correction
        checker = detector.getBestColorChecker()
        chartsRGB = checker.getChartsRGB()

        src = chartsRGB[:, 1].copy().reshape(24, 1, 3)
        src /= 255.0

        model = cv2.ccm_ColorCorrectionModel(
            src, imUtils.chartsRGB_np, cv2.ccm.COLOR_SPACE_sRGB)

        model.setWeightCoeff(1)

        model.run()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float64) / 255
        calibrated = model.infer(img_rgb)
        calibrated = imUtils.toOpenCVU8(calibrated.copy())

        patchPos = imUtils.getCardsPos(img.copy())

        filled, cnts = imUtils.masking(calibrated.copy(),logger,err_list, file)

        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
        for pos in patchPos.values():
            (x, y, w, h) = pos
            
        cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))
        try:
            max_cnt = max(cnts, key=cv2.contourArea)
        except:
            print("Cnt contains no value")
            imUtils.log_err(logger, msg=f'{file}: Cnt contains no value')
            imUtils.append_err_list(err_list, file)
            return

        x, y, w, h = cv2.boundingRect(max_cnt)
        img = img[y:y+h, x:x+w]

    else:
        patchPos, EXTRACTED_RGB, REF_RGB = imUtils.get4PatchInfo(img.copy())

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

        # currently the bestprint("corrected Vandermonde:")
        calibrated = colour.colour_correction(
            img, EXTRACTED_RGB, REF_RGB, 'Vandermonde')
        img = imUtils.toOpenCVU8(calibrated.copy())
        _, cnts = imUtils.masking(img.copy(),logger,err_list, file)

        cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))

        # checking if max() arg is empty also filter out the unqualified images (e.g. ones with no colorChecker)
        try:
            max_cnt = max(cnts, key=cv2.contourArea)
        except:
            print("Cnt contains no value")
            imUtils.log_err(logger, msg=f'{file}: Cnt contains no value')
            imUtils.append_err_list(err_list,file)
            return

        x, y, w, h = cv2.boundingRect(max_cnt)
        img = img[y:y+h, x:x+w]

    # Scales kernel size by scaling factor computed for better masking
    kernel_size_scaled = math.floor(5 * scaling_factor)

    filled, max_cnt = imUtils.masking(
        img.copy(), logger, err_list, file, kernel_size_scaled, 'biggest')
    if filled is None and max_cnt is None:
        print("retuned values from making() is none")
        return
    x, y, w, h = cv2.boundingRect(max_cnt)

    # TODO: crop 1000x500 centered on the above max_cnt

    mask = filled[imUtils.MARGIN:y+h-imUtils.MARGIN,
                  imUtils.MARGIN:x+w-imUtils.MARGIN]
    img = img[imUtils.MARGIN:y+h-imUtils.MARGIN,
              imUtils.MARGIN:x+w-imUtils.MARGIN]

    sub_imgs = []

    h, w = img.shape[0], img.shape[1]

    for i in range(5000):
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
    if len(sub_imgs) == 0:
        print('nth found!')
        imUtils.log_err(logger, msg=f'STATUS - {file} has no cropped sherd found')
        imUtils.append_err_list(err_list,file)
        return None
    else:
        imUtils.log_err(logger, msg=f'STATUS - {file}: SUCCESS')
        return sub_imgs
