import sys
sys.path.append('../')

from pathlib import Path
import splitfolders
import configure as cfg

def splitDataset(): 
    # Split the dataset after processed with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    Path(cfg.SPLITTED_DIR).mkdir(parents=True, exist_ok=True)
    splitfolders.ratio(cfg.PROCESSED_DIR, output=cfg.SPLITTED_DIR,
        seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # If move=True, images will be moved from processed_images instead of copy

    # If empty folders in validation set, remove them from train and test folders as well
    for dirpath, dirnames, files in os.walk(cfg.SPLITTED_DIR+"/val/"):
            if not files:
                print("Diretory {0} is empty".format(dirpath))
                if not os.system(f'rm -d {dirpath}'):
                    print(f"deleted successfully {dirpath}")
                    os.system(f'rm -r {cfg.SPLITTED_DIR+"/test/" + dirpath.split(os.path.sep)[-1]}')
                    print(f"deleted test - {dirpath.split(os.path.sep)[-1]}")
                    os.system(f'rm -r {cfg.SPLITTED_DIR+"/train/" + dirpath.split(os.path.sep)[-1]}')
                    print(f"deleted train - {dirpath.split(os.path.sep)[-1]}")