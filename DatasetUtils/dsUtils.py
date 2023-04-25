import sys
sys.path.append('../')

import os
from pathlib import Path
import splitfolders
import configure as cfg
import pandas as pd
import os
import shutil
import seaborn as sns
import plotly.express as px
import argparse
# import imUtils 
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

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
    num_samples_dict = {'Class Code':[], 'Count': []}
    for dirpath, dirnames, filenames in os.walk(file_path):
        dirnames.sort(key=lambda s: float('inf') if s == 'unlabeled' else int(s))
        N_class += len(dirnames)
        N_f = len(filenames)
        N += N_f
        if str(Path(dirpath).name).isdigit():
            num_samples_dict['Class Code'].append(Path(dirpath).name)
            num_samples_dict['Count'].append(N_f)
        if isDetailed and dirpath != str(file_path): print(f"Number of samples in Class {Path(dirpath).name}: {N_f}" )
    print(f"Total Files {N} with {N_class} classes")
    
    return pd.DataFrame(num_samples_dict)

def generateDatasetByFeature(targetFolder, by):
    print('generating')
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

def raw2jpg(src, dst): 
    img = imUtils.imread(src)
    img = img[:,:,::-1]
    img = Image.fromarray(img)
    img.save(dst)

def genNiceBarChart(df, by):
    fig, ax = plt.subplots(figsize=(13.33,7.5), dpi = 96)
    # plt.tight_layout(pad=1)
    bar1 = ax.bar(df['Class Code'], df['Count'], width=0.6)

    # Create the grid 
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('', fontsize=12, labelpad=10) # No need for an axis label
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12, labelrotation=0)
    df_Mapping = pd.read_csv(f'../Labelling/{by}LabelEncodingMapping.csv') 
    labels = df_Mapping['Class'].tolist()
    labels.pop()
    ax.set_xticks(df['Class Code'], labels) # Map integers numbers from the series to labels list

    # Reformat y-axis
    ax.set_ylabel('Number of Sherd Images', fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

    # Add label on top of each bar
    ax.bar_label(bar1, labels=[f'{e}' for e in df['Count']], padding=3, color='black', fontsize=8) 
    
    # Remove the spines
    ax.spines[['top','left','bottom']].set_visible(False)

    # Make the left spine thicker
    ax.spines['right'].set_linewidth(1.1)

    # Add in red line and rectangle on top
    ax.plot([0.12, .9], [.98, .98], transform=fig.transFigure, clip_on=False, color='#E3120B', linewidth=.6)
    ax.add_patch(plt.Rectangle((0.12,.98), 0.04, -0.02, facecolor='#E3120B', transform=fig.transFigure, clip_on=False, linewidth = 0))

    # Add in title and subtitle
    ax.text(x=0.12, y=.93, s="Image data distribution per Color Class", transform=fig.transFigure, ha='left', fontsize=14, weight='bold', alpha=.8)
    # ax.text(x=0.12, y=.90, s="Difference in minutes between scheduled and actual arrival time averaged over each Class Code", transform=fig.transFigure, ha='left', fontsize=12, alpha=.8)

    # Set source text
    # ax.text(x=0.1, y=0.12, s="Source: Kaggle - Airlines Delay - https://www.kaggle.com/datasets/giovamata/airlinedelaycauses", transform=fig.transFigure, ha='left', fontsize=10, alpha=.7)

    # Adjust the margins around the plot area
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    # Set a white background
    fig.patch.set_facecolor('white')
    
    # Colours - Choose the extreme colours of the colour map
    colours = ["#2196f3", "#bbdefb"]

    # Colormap - Build the colour maps
    cmap = mpl.colors.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
    norm = mpl.colors.Normalize(df['Count'].min(), df['Count'].max()) # linearly normalizes data into the [0.0, 1.0] interval

    # Plot bars
    bar1 = ax.bar(df['Class Code'], df['Count'], color=cmap(norm(df['Count'])), width=0.6, zorder=2)
        
    # Find the average data point and split the series in 2
    average = df['Count'].mean()
    below_average = df[df['Count']<average]
    above_average = df[df['Count']>=average]
    
    # Colours - Choose the extreme colours of the colour map
    colors_high = ["#ff5a5f", "#c81d25"] # Extreme colours of the high scale
    colors_low = ["#2196f3","#bbdefb"] # Extreme colours of the low scale

    # Colormap - Build the colour maps
    cmap_low = mpl.colors.LinearSegmentedColormap.from_list("low_map", colors_low, N=256)
    cmap_high = mpl.colors.LinearSegmentedColormap.from_list("high_map", colors_high, N=256)
    norm_low = mpl.colors.Normalize(below_average['Count'].min(), average) # linearly normalizes data into the [0.0, 1.0] interval
    norm_high = mpl.colors.Normalize(average, above_average['Count'].max())

    # Plot bars and average (horizontal) line
    bar1 = ax.bar(below_average['Class Code'], below_average['Count'], color=cmap_low(norm_low(below_average['Count'])), width=0.6, label='Below Average', zorder=2)
    bar2 = ax.bar(above_average['Class Code'], above_average['Count'], color=cmap_high(norm_high(above_average['Count'])), width=0.6, label='Above Average', zorder=2)
    plt.axhline(y=average, color = 'grey', linewidth=3)

    # Determine the y-limits of the plot
    ymin, ymax = ax.get_ylim()
    # Calculate a suitable y position for the text label
    y_pos = average/ymax + 0.03
    # Annotate the average line
    ax.text(0.88, y_pos, f'Average = {average:.1f}', ha='right', va='center', transform=ax.transAxes, size=8, zorder=3)

    # Add legend
    ax.legend(loc="best", ncol=2, bbox_to_anchor=[1, 1.07], borderaxespad=0, frameon=False, fontsize=8)
    
    fig.savefig(f'{by}_beauti_count.jpg')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--by', type=str, default='color', help='options: color, texture, texture2')
    FLAGS = parser.parse_args()

    by = FLAGS.by
    dir = 'processed_images' + ('' if by == 'detailed' else f'_by_{by}') 
    processed_data_dir = cfg.DATA_DIR / dir
    # generateDatasetByFeature(processed_data_dir, by)
    # print("Splitting processed dataset")
    # splitted_data_dir = cfg.DATA_DIR / ('splitted_' + dir)
    # splitDataset(processed_data_dir, splitted_data_dir) 

    # generateDatasetByFeature(targetFolder, by)
    # root = '/userhome/2072/fyp22007/data/processed_images/unlabeled'
    # file = '478130_4419430_37_33_1_s1.jpg'
    # path = Path(root) / file
    # print(f'path: {path}')
    # dir = path.parent.name

    # parentImageName = "_".join((file.split('_'))[0:4]) # the original image in raw image dataset
    # side = file.split('_')[4] # Front or back side of a sherd 
    # print(f'dir: {dir}')
    # print(f'file: {file}')
    # print(parentImageName)

    # if dir != "unlabeled": # Ignore the unlabeled class
    #     print(df[df['file_name'] == parentImageName][code].values[0])
    #     FolderCode = str(df[df['file_name'] == parentImageName][code].values[0])
    #     targetFolderCode = targetFolder / FolderCode
    #     targetFolderCode.mkdir(parents=True, exist_ok=True)
    #     shutil.copy(originFolder / dir / file, targetFolderCode / file)

    # Count number of data
    # by = 'total'
    if by == 'total':
        df = countSamplesGenerated(cfg.DATA_DIR / f'processed_images', True)
    else: 
        df = countSamplesGenerated(cfg.DATA_DIR / f'processed_images_by_{by}', True)
    genNiceBarChart(df,by)
    # fig = px.bar(df, x='Class Code', y='Count')
    # fig.write_image(f"{by}_count.jpg")
    # fig.write_image(f'cfg.DATA_DIR/{by}_count.jpg')
