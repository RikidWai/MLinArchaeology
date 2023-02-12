import sys
sys.path.append('../')

import os
from pathlib import Path
import splitfolders
import configure as cfg

def splitDataset(): 
    # Split the dataset after processed with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    cfg.SPLITTED_DIR.mkdir(parents=True, exist_ok=True)
    splitfolders.ratio(cfg.PROCESSED_DIR, output=cfg.SPLITTED_DIR,
        seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # If move=True, images will be moved from processed_images instead of copy

    # If empty folders in validation set, remove them from train and test folders as well
    for dirpath, dirnames, files in os.walk(cfg.SPLITTED_DIR / "/val/"):
            if not files:
                print("Diretory {0} is empty".format(dirpath))
                if not os.system(f'rm -d {dirpath}'):
                    print(f"deleted successfully {dirpath}")
                    os.system(f'rm -r {cfg.SPLITTED_DIR / "/test/" / dirpath.split(os.path.sep)[-1]}')
                    print(f"deleted test - {dirpath.split(os.path.sep)[-1]}")
                    os.system(f'rm -r {cfg.SPLITTED_DIR / "/train/" / dirpath.split(os.path.sep)[-1]}')
                    print(f"deleted train - {dirpath.split(os.path.sep)[-1]}")
                    
                    
def rm_tree(pth):
    pth = Path(pth)
    if pth.is_dir():
        for child in pth.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
    else: 
        print('No Folder is found')
        
def countSamplesGenerated(file_path, isDetailed = False):
    N = 0  # total files
    N_class = 0 # total classes
    for dirpath, dirnames, filenames in os.walk(file_path):
        dirnames.sort(key=lambda s: float('inf') if s == 'unlabeled' else int(s))
        N_class += len(dirnames)
        N_f = len(filenames)
        N += N_f
        if isDetailed and dirpath != str(file_path): print(f"Number of samples in Class {Path(dirpath).name}: {N_f}" )
    print(f"Total Files {N} with {N_class} classes")
    
    
    return N

if __name__ == '__main__':
    countSamplesGenerated(cfg.PROCESSED_DIR, True)
    
    # print("Splitting processed dataset")
    splitDataset() 