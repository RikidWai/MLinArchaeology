import sys
sys.path.append('../')

import os
from pathlib import Path
import splitfolders
import configure as cfg

import os
import shutil
import pandas as pd
from pathlib import Path

def splitDataset(processed_data_dir, splitted_data_dir): 
    # Split the dataset after processed with a ratio.
    # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
    

    
    splitted_data_dir.mkdir(parents=True, exist_ok=True)
    splitfolders.ratio(processed_data_dir, output=splitted_data_dir,
        seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # If move=True, images will be moved from processed_images instead of copy

    # If empty folders in validation set, remove them from train and test folders as well
    for dirpath, dirnames, files in os.walk(splitted_data_dir / "val"):
        if not any(Path(dirpath).iterdir()):
            print("Diretory {0} is empty".format(dirpath))
            if not os.system(f'rm -d {dirpath}'):
                print(f"deleted successfully {dirpath}")
                os.system(f'rm -r {splitted_data_dir / "test/" / dirpath.split(os.path.sep)[-1]}')
                print(f"deleted test - {dirpath.split(os.path.sep)[-1]}")
                os.system(f'rm -r {splitted_data_dir / "train/" / dirpath.split(os.path.sep)[-1]}')
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

def generateDatasetByFeature(targetFolder, by):
    
    filepath = cfg.MAIN_DIR / "Labelling/labelEncoding.csv"
    df = pd.read_csv(filepath)

    originFolder = cfg.PROCESSED_DIR
    code = by + '_code'
    
    for root, dirs, files in os.walk(originFolder):
        dirs.sort()
        for file in files:
            path = Path(root) / file
            dir = path.parent.name
            parentImageName = "_".join((file.split('_'))[0:4]) # the original image in raw image dataset
            side = file.split('_')[4] # Front or back side of a sherd 
            
            # if dir == "unlabeled":
            #     if not os.path.exists(targetFolder / 'unlabeled'):
            #         os.makedirs(targetFolder / 'unlabeled')
            #     shutil.copy(f'{originFolder / dir }/'+file, f'{targetFolder / "unlabeled"}/'+file)
            if dir !=  "unlabeled" and side == '2': # Ignore the unlabeled class and select the back sides only
                if not df[df['file_name'] == parentImageName][by].isnull().values.any():
                    FolderCode = str(df[df['file_name'] == parentImageName][code].values[0])
                    targetFolderCode = targetFolder / FolderCode
                    targetFolderCode.mkdir(parents=True, exist_ok=True)
                    shutil.copy(originFolder / dir / file, targetFolderCode / file)


# def generateColorDatabase():
#     filepath = cfg.MAIN_DIR / "Labelling/labelEncoding.csv"
#     df = pd.read_csv(filepath)

#     targetFolder = cfg.DATA_DIR / "processed_images_by_color/"
#     originFolder = cfg.PROCESSED_DIR

#     for root, dirs, files in os.walk(originFolder):
#         dirs.sort()
#         for file in files:
#             path = os.path.join(root, file)
#             dir = root.split(os.path.sep)[-1]
#             print(dir)
#             parentImageName = "_".join((file.split('_'))[0:4])
#             print(parentImageName)
#             if dir == "unlabeled":
#                 if not os.path.exists(targetFolder / 'unlabeled'):
#                     os.makedirs(targetFolder / 'unlabeled')
#                 shutil.copy(f'{originFolder / dir }/'+file, f'{targetFolder / "unlabeled"}/'+file)
#             else:
#                 colorFolderCode = df[df['file_name'] == parentImageName]['color_code'].values[0]
                
#                 if not os.path.exists(targetFolder / str(colorFolderCode)):
#                     os.makedirs(targetFolder / str(colorFolderCode))
#                 shutil.copy(f'{originFolder / dir }/'+file, f'{targetFolder / str(colorFolderCode)}/'+file)

# def generateTextureDatabase():
#     filepath = cfg.MAIN_DIR / "Labelling/labelEncoding.csv"
#     df = pd.read_csv(filepath)

#     targetFolder = cfg.DATA_DIR / "processed_images_by_texture/"
#     originFolder = cfg.PROCESSED_DIR

#     for root, dirs, files in os.walk(originFolder):
#         dirs.sort()
#         for file in files:
#             path = os.path.join(root, file)
#             dir = root.split(os.path.sep)[-1]
#             print(dir)
#             parentImageName = "_".join((file.split('_'))[0:4])
#             print(parentImageName)
#             if dir == "unlabeled":
#                 if not os.path.exists(targetFolder / 'unlabeled'):
#                     os.makedirs(targetFolder / 'unlabeled')
#                 shutil.copy(f'{originFolder / dir }/'+file, f'{targetFolder / "unlabeled"}/'+file)
#             else:
#                 textureFolderCode = df[df['file_name'] == parentImageName]['texture_code'].values[0]
                
#                 if not os.path.exists(targetFolder / str(textureFolderCode)):
#                     os.makedirs(targetFolder / str(textureFolderCode))
#                 shutil.copy(f'{originFolder / dir }/'+file, f'{targetFolder / str(textureFolderCode)}/'+file)

if __name__ == '__main__':
    
    filepath = cfg.MAIN_DIR / "Labelling/labelEncoding.csv"
    df = pd.read_csv(filepath)

    targetFolder = cfg.DATA_DIR / 'processed_images_by_texture'
    
    by = 'texture'
    originFolder = cfg.PROCESSED_DIR
    code = by + '_code'
    root = '/userhome/2072/fyp22007/data/processed_images/unlabeled'
    file = '478130_4419430_37_33_1_s1.jpg'
    path = Path(root) / file
    print(f'path: {path}')
    dir = path.parent.name

    parentImageName = "_".join((file.split('_'))[0:4]) # the original image in raw image dataset
    side = file.split('_')[4] # Front or back side of a sherd 
    print(f'dir: {dir}')
    print(f'file: {file}')
    print(parentImageName)

    if dir != "unlabeled": # Ignore the unlabeled class
        print(df[df['file_name'] == parentImageName][code].values[0])
        FolderCode = str(df[df['file_name'] == parentImageName][code].values[0])
        targetFolderCode = targetFolder / FolderCode
        targetFolderCode.mkdir(parents=True, exist_ok=True)
        shutil.copy(originFolder / dir / file, targetFolderCode / file)

    
    # generateColorDatabase()
    # generateTextureDatabase()
    
    # countSamplesGenerated(cfg.PROCESSED_DIR, True)
    
    # # print("Splitting processed dataset")
    # splitDataset() 