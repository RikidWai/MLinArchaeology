import sys
sys.path.append('../')

import os
import pandas as pd
import configure as cfg
from imProcessingPipeline import improcessing as process
import imUtils
import cv2
df_encoding = pd.read_csv('../Labelling/LabelEncoding.csv')


def main(argv):
    # For loggging errors
    logger = imUtils.init_logger()
    err_list = []

    # Looping begins
    for root, dirs, files in os.walk(cfg.DATA_DIR):
        for file in files:
            filename, extension = os.path.splitext(file)
            if 'cr' in extension or 'CR' in extension:
                path = os.path.join(root, file)
                dir = root.split(os.path.sep)[-1]
                print(f'')
                subImgs = process(path, logger, err_list)
                # if subImgs != None:
                #     # check the dir match filename column in LabelEncoding, then put into respective folder
                #     print(dir)
                #     targetFolder = str(df_encoding.query(
                #         f'file_name == "{dir}"').fabric)
                #     print(cfg.TARGET_DIR + targetFolder + os.path.sep)
                #     print('fuck')
                #     if not os.path.exists(targetFolder):
                #         os.makedirs(targetFolder)
                #     for i, sub_img in enumerate(subImgs):
                #         cv2.imwrite(f'{dir}_{filename}_s{i+1}.jpg', sub_img)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        print('Usage: imProcessing.py [, dst_ppc [, cropped_dim ] ]')
        sys.exit(1)
    main(sys.argv)

    