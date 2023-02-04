import sys
sys.path.append('../')

import os
import pandas as pd
import configure as cfg
from imProcessingPipeline import improcessing as process
import imUtils
import cv2
from pathlib import Path
from Labelling.labelling import generateEncoding
from DatasetUtils import dsUtils

    
def main(argv):
    generateEncoding()
    df_encoding = pd.read_csv('../Labelling/labelEncoding.csv')
    df_encoding.drop(df_encoding.filter(regex="Unname"),axis=1, inplace=True)
    

    total = 0
    num_success = 0 
    num_samples = 0
    # For loggging errors
    logger = imUtils.init_logger()
    err_list = []
    
    # confirm = input("Process image will delete the previous samples. Are you sure(y/n)? ")
    # if confirm == 'y':
    #     rm_tree(cfg.PROCESSED_DIR)
    #     rm_tree(cfg.SPLITTED_DIR)
    #     print('Previous results are deleted. Now start to process.')
    # else: 
    #     print('abort')
    #     return
    
    # dsUtils.rm_tree(cfg.PROCESSED_DIR)
    # dsUtils.rm_tree(cfg.SPLITTED_DIR)
    # print('Previous results are deleted. Now start to process.')
    
    Path(cfg.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    # Looping begins
    for root, dirs, files in os.walk(cfg.RAWIMG_DIR):
        dirs.sort()
        for file in files:
            print(Path(root) / Path(file))
            filename, extension = os.path.splitext(file)
            if 'cr' in extension or 'CR' in extension:
                total += 1 
                path = os.path.join(root, file)
                dir = root.split(os.path.sep)[-1]
                if dir > "478130_4419430_3_212":
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
                            
                            targetFolder = str(targetFolder.iloc[0]['fabric']) if targetFolder.empty != True else 'unlabeled'
                            
                            if not os.path.exists(cfg.PROCESSED_DIR / targetFolder):
                                os.makedirs(cfg.PROCESSED_DIR / targetFolder)
                            
                            num_success +=1 
                            num_samples += len(subImgs)
                            for i, sub_img in enumerate(subImgs):           
                                cv2.imwrite(f'{cfg.PROCESSED_DIR / targetFolder/ dir}_{filename}_s{i+1}.jpg', sub_img)
                        else:
                            imUtils.log_err(logger, msg=f'Image not in the correct directory structure {path}') 
    imUtils.err_list_to_csv(err_list)
    msg = f'Total {total} images are processed, \
          {num_success} labeled images output {num_samples} samples'
    
    imUtils.log_err(logger, msg=msg) 
    print(msg)

    # dsUtils.splitDataset()

                            

if __name__ == '__main__':
    main(sys.argv)

    