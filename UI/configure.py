from pathlib import Path

MAIN_DIR = Path('/userhome/2072/fyp22007/MLinAraechology/') # the source code path 
DATA_DIR = Path('/userhome/2072/fyp22007/data/') # the directory storing all images 

RAWIMG_DIR = DATA_DIR / 'raw_images/' # the directory storing the raw images 
PROCESSED_DIR = DATA_DIR / 'processed_images/' # the directory storing the processed images 
SPLITTED_DIR = DATA_DIR / 'splitted_processed_images' # the directory storing the train/test images

# Image Processing Parameters
MAX_WIDTH = 256 # Dimension of cropped img
MAX_HEIGHT = 256
SAMPLE_NUM = 6  # num of cropped img cropped from raw img
DST_PPC = 256 # pixel per cm 