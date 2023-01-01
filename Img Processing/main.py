import sys
sys.path.append('../')

import os
import pandas as pd
import configure as cfg
from imProcessingPipeline import improcessing as process
import imUtils
import cv2
df_encoding = pd.read_csv('../Labelling/LabelEncoding.csv')
df_encoding.drop(df_encoding.filter(regex="Unname"),axis=1, inplace=True)

def main(argv):
    # For loggging errors
    logger = imUtils.init_logger()
    err_list = []

    # Looping begins
    for root, dirs, files in os.walk(cfg.DATA_DIR):
        dirs.sort()
        for file in files:
            filename, extension = os.path.splitext(file)
            if 'cr' in extension or 'CR' in extension:
                path = os.path.join(root, file)
                dir = root.split(os.path.sep)[-1]

                try:
                    subImgs = process(path, logger, err_list)
                except Exception as e:
                    imUtils.log_err(logger, msg=f'{path}: Cant process image. imProcessing HAS BUG. Exception {e}') 
                    imUtils.append_err_list(err_list, path)
                    continue

                if subImgs != None:
                    # check the dir match filename column in LabelEncoding, then put into respective folder
                    if len(dir)>0:
                        targetFolder = df_encoding.query(
                            f'file_name == "{dir}"')
                        if targetFolder.empty != True:
                            targetFolder = str(targetFolder.iloc[0]['fabric'])

                            if not os.path.exists(cfg.TARGET_DIR + targetFolder):
                                os.makedirs(cfg.TARGET_DIR + targetFolder)

                            for i, sub_img in enumerate(subImgs):
                                cv2.imwrite(f'{cfg.TARGET_DIR}{targetFolder}/{dir}_{filename}_s{i+1}.jpg', sub_img)
                        else:
                            imUtils.log_err(logger, msg=f'no target label found for this image {path}') 
                    else:
                        imUtils.log_err(logger, msg=f'Image not in the correct directory structure {path}') 
    imUtils.err_list_to_csv(err_list)
                            

if __name__ == '__main__':
    if len(sys.argv) > 3:
        print('Usage: imProcessing.py [, dst_ppc [, cropped_dim ] ]')
        sys.exit(1)
    main(sys.argv)

    