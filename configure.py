from pathlib import Path

MAIN_DIR = Path('/userhome/2072/fyp22007/MLinAraechology/')
DATA_DIR = Path('/userhome/2072/fyp22007/data/')

RAWIMG_DIR = DATA_DIR / 'raw_images/'
PROCESSED_DIR = DATA_DIR / 'processed_images/' 
SPLITTED_DIR = DATA_DIR / 'splitted_processed_images'


# Image Processing Parameters
MAX_WIDTH = 256 # Dimension of cropped img
MAX_HEIGHT = 256
SAMPLE_NUM = 6  # num of cropped img cropped from raw img
DST_PPC = 256 # pixel per cm 