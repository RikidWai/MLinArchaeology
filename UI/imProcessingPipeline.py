import sys
sys.path.append('../')

import math
import colour
import configure as cfg
import imUtils
import cv2
import numpy as np
import timeit

detector = cv2.mcc.CCheckerDetector_create()

def improcessing(img):
    # imUtils.imshow(img)
    img = imUtils.white_bal(img)
    # imUtils.imshow(img,'ori')
    # Convert the format for color correction
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255
    
    # Color Correction
    is24Checker = imUtils.detect24Checker(img.copy(), detector)
    
    if is24Checker:
        try:
            patchPos = imUtils.getCardsBlackPos(img.copy())
        except Exception as e:
            raise ValueError(f'Error getting patch positions: {e}')
        
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
    # imUtils.imshow(img,'calibrated')
    # Scale image according to required ppc ratio
    try:
        img, scaling_factor = imUtils.scale_img(img.copy(), cfg.DST_PPC, patchPos)
    except Exception as e:
        raise ValueError(f'Error scaling image: {e}')

    # imUtils.imshow(img,'scaled')

    # Scales kernel size by scaling factor computed for better masking
    kernel_size = 6 if max(img.shape) >= 1000 else 5

    # Scales kernel size by scaling factor computed for better masking
    filled, cnts = imUtils.masking(img.copy(), kernel_size)
    # imUtils.imshow(filled, 'filled')

    
    
    try: 
        sherd_cnt = imUtils.getSherdCnt(img.copy(), cnts, is24Checker)
    except:
        raise ValueError("Fail to get feasible sherd")

    x, y, w, h = cv2.boundingRect(sherd_cnt)
    mask = filled[y:y+h, x:x+w]
    img = img[y:y+h, x:x+w]

    # imUtils.imshow(mask, 'mask')
    # imUtils.imshow(img, 'sherd')

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
        raise ValueError('nth found!')
    return sub_imgs

        
if __name__ == '__main__':
    # To process one single image
    
    start = timeit.default_timer()
    img = imUtils.imread('/userhome/2072/fyp22007/MLinAraechology/test_images/1.CR2')
    sub_imgs = improcessing(img)
    # sub_imgs = improcessing('/userhome/2072/fyp22007/data/raw_images/478130_4419430_3_48/1.CR2', logger, err_list)
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    if sub_imgs is not None: 
        imUtils.imshow(sub_imgs[0], 'result')
        # cv2.imwrite(f'../test_images/test.jpg', sub_imgs[0])
