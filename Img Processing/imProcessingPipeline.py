import sys
sys.path.append('../')

import numpy as np
import cv2
import imUtils
import configure as cfg
import colour
import math
import os

ROOT_DIR = os.getcwd()
DEFAULT_DST_PPC = 118
DEFAULT_OUT_DIM = (1000, 500)


def improcessing(file, logger, err_list):
    print("filename: ",file)
    dst_ppc = DEFAULT_DST_PPC  # Default value
    out_w, out_h = DEFAULT_OUT_DIM

    print(f'Using dst_ppc {dst_ppc}')


    img = imUtils.imread(file)
    print('shape', img.shape)

    img = imUtils.white_bal(img)
    detector = cv2.mcc.CCheckerDetector_create()
    scaling_factor = 1
    # Scale image according to required ppc ratio
    try:
        img, scaling_factor = imUtils.scale_img(img, dst_ppc)
    except Exception as e:
        print(f'Error scaling image: {e}')
        imUtils.log_err(logger, err=e)
        imUtils.append_err_list(err_list,file)
        return

    imUtils.display_image(img, 'see scaled')
    cv2.imwrite('../out_images/img_scaled_1.jpeg', img)
    img2 = img.copy()

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
        calibrated = model.infer(img_rgb)
        calibrated = imUtils.toOpenCVU8(calibrated.copy())

        patchPos = imUtils.getCardsPos(img.copy())

        filled, cnts = imUtils.masking(calibrated.copy(),logger,err_list, file)

        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 5)
        print('pos', patchPos)
        for pos in patchPos.values():
            (x, y, w, h) = pos
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 5)
        print('shape', img.shape, img2.shape)
        imUtils.imshow(img2, 'copy')
        cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))
        try:
            max_cnt = max(cnts, key=cv2.contourArea)
        except:
            print("Cnt contains no value")
            imUtils.log_err(logger, msg=f'{file}: Cnt contains no value')
            imUtils.append_err_list(err_list, file)
            return

        x, y, w, h = cv2.boundingRect(max_cnt)
        img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
                  x-imUtils.MARGIN:x+w+imUtils.MARGIN]

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
        img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
                  x-imUtils.MARGIN:x+w+imUtils.MARGIN]

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
    if len(sub_imgs) == 0:
        print('nth found!')
        imUtils.log_err(logger, msg=f'STATUS - {file} has no cropped sherd found') #need add this to err list now
        imUtils.append_err_list(err_list,file)
        return None
    else:
        imUtils.log_err(logger, msg=f'STATUS - {file}: SUCCESS')
        imUtils.imshow(sub_imgs[0])
        return sub_imgs

    # Save the cropped regions
    # for i, sub_img in enumerate(sub_imgs):
    #     imUtils.imshow(sub_img)
    # cv2.imwrite(f'{i + 1}.jpg', sub_img)
