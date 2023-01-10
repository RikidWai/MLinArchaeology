import sys
sys.path.append('../')

import os
import math
import colour
import configure as cfg
import imUtils
import cv2
import numpy as np



DEFAULT_OUT_DIM = (1000, 500)


def improcessing(file, logger, err_list):
    out_w, out_h = DEFAULT_OUT_DIM
    img = imUtils.imread(file)

    img = imUtils.white_bal(img)
    imUtils.imshow(img)
    detector = cv2.mcc.CCheckerDetector_create()

    # Scale image according to required ppc ratio
    try:
        img, scaling_factor = imUtils.scale_img(img, cfg.DST_PPC)
    except Exception as e:
        print(f'Error scaling image: {e}')
        imUtils.log_err(logger, err=e)
        imUtils.append_err_list(err_list, file)
        return

    # Convert the format for color correction
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    
    if imUtils.detect24Checker(img.copy(), detector):
        # Color Correction
        patchPos = imUtils.getCardsPos(img.copy())
        checker = detector.getBestColorChecker()
        chartsRGB = checker.getChartsRGB()

        src = chartsRGB[:, 1].copy().reshape(24, 1, 3) / 255.0

        model = cv2.ccm_ColorCorrectionModel(
            src, imUtils.chartsRGB_np, cv2.ccm.COLOR_SPACE_sRGB)

        model.setWeightCoeff(1)

        model.run()
        calibrated = model.infer(rgb)

    else:
        patchPos, EXTRACTED_RGB, REF_RGB = imUtils.get4PatchInfo(img.copy())
        # currently the bestprint("corrected Vandermonde:")
        calibrated = colour.colour_correction(
            rgb, EXTRACTED_RGB, REF_RGB, 'Vandermonde')

    img = imUtils.toOpenCVU8(calibrated.copy())
    
    # Scales kernel size by scaling factor computed for better masking
    kernel_size_scaled = math.floor(5 * scaling_factor)
    
    filled, cnts = imUtils.masking(img.copy(), logger, err_list, file, kernel_size = kernel_size_scaled)

    cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))

    # checking if max() arg is empty also filter out the unqualified images (e.g. ones with no colorChecker)
    try:
        max_cnt = max(cnts, key=cv2.contourArea)
    except:
        print("Cnt contains no value")
        imUtils.log_err(logger, msg=f'{file}: Cnt contains no value')
        imUtils.append_err_list(err_list, file)
        return

    x, y, w, h = cv2.boundingRect(max_cnt)
    mask = filled[y:y+h, x:x+w]
    img = img[y:y+h, x:x+w]

    # TODO: crop 1000x500 centered on the above max_cnt

    imUtils.imshow(mask, 'mask')
    imUtils.imshow(img, 'sherd')
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
        imUtils.log_err(
            logger, msg=f'STATUS - {file} has no cropped sherd found')
        imUtils.append_err_list(err_list, file)
        return None
    else:
        imUtils.log_err(logger, msg=f'STATUS - {file}: SUCCESS')
        return sub_imgs


if __name__ == '__main__':
    # For loggging errors
    logger = imUtils.init_logger()
    err_list = []

    sub_imgs = improcessing('../test_images/1.cr2', logger, err_list)
    if sub_imgs is not None: 
        imUtils.imshow(sub_imgs[0])
