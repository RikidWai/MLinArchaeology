import numpy as np
import cv2
import imUtils
import configure as cfg
import colour
import sys
import math


DEFAULT_DST_PPC = 118
DEFAULT_OUT_DIM = (1000, 500)


def improcessing(file):

    # if len(argv) > 1:
    #     dst_ppc = float(argv[1])
    #     if dst_ppc <= 0:
    #         print('dst_ppc must be positive')
    #         dst_ppc = DEFAULT_DST_PPC
    #     if len(argv) > 2:  # For cropping as output
    #         out_w, out_h = int(argv[2][0]), int(argv[2][1])
    #     else:
    #         out_w, out_h = DEFAULT_OUT_DIM
    # else:
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
        sys.exit(1)
    img2 = img.copy()
    # -- Tests the effect of scaling on final sherd mask --
    # Higher scaling factor leads to a more disconnected mask
    # Try adjusting kernel size according to input image dimension to solve this
    # scaling_factor = 1.2
    # img = cv2.resize(img.copy(), None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    # cv2.imwrite('output_scaled.jpeg', img)

    # Testing only above
    # if True:
    #     return

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

        patchPos = imUtils.getCardsPos(img.copy())

        filled, cnts = imUtils.masking(calibrated.copy())

        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 255, 0), 5)
        print('pos', patchPos)
        for pos in patchPos.values():
            print('this is pos', pos)
            (x, y, w, h) = pos
            cv2.rectangle(img2, (x, y), (x+w, y+h), (0, 0, 255), 5)
        print('shape', img.shape, img2.shape)
        imUtils.imshow(img2, 'copy')
        cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))
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

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

        # currently the bestprint("corrected Vandermonde:")
        calibrated = colour.colour_correction(
            img, EXTRACTED_RGB, REF_RGB, 'Vandermonde')
        img = imUtils.toOpenCVU8(calibrated.copy())
        _, cnts = imUtils.masking(img.copy())

        cnts = list(filter(lambda cnt: imUtils.isSherd(cnt, patchPos), cnts))

        # checking if max() arg is empty also filter out the unqualified images (e.g. ones with no colorChecker)
        try:
            max_cnt = max(cnts, key=cv2.contourArea)
        except:
            print("Cnt contains no value")
            sys.exit(1)
        x, y, w, h = cv2.boundingRect(max_cnt)
        img = img[y-imUtils.MARGIN:y+h+imUtils.MARGIN,
                  x-imUtils.MARGIN:x+w+imUtils.MARGIN]

    # Scales kernel size by scaling factor computed for better masking
    kernel_size_scaled = math.floor(5 * scaling_factor)

    filled, max_cnt = imUtils.masking(
        img.copy(), kernel_size_scaled, 'biggest')
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
        return None
    else:
        imUtils.imshow(sub_imgs[0])
        return sub_imgs

    # Save the cropped regions
    # for i, sub_img in enumerate(sub_imgs):
    #     imUtils.imshow(sub_img)
    # cv2.imwrite(f'{i + 1}.jpg', sub_img)
