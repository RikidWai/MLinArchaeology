# MLinArchaeology

### Setup and Configuration

1. Run the following code

```
# Setup conda environment
conda create -n archaeoML
conda activate archaeoML
# pip list --format=freeze > requirements.txt

# Setup dependencies
pip install -r requirements.txt
```

2. Configure your paths for the data and the code. The default path is as follows:

```
MAIN_DIR = Path('/userhome/2072/fyp22007/MLinAraechology/') # the source code path
DATA_DIR = Path('/userhome/2072/fyp22007/data/') # the directory storing all images

RAWIMG_DIR = DATA_DIR / 'raw_images/' # the directory storing the raw images
PROCESSED_DIR = DATA_DIR / 'processed_images/' # the directory storing the processed images
SPLITTED_DIR = DATA_DIR / 'splitted_processed_images' # the directory storing the train/test images
```

3. Configure parameters for image processing. The default values are as follows:

```
MAX_WIDTH = 256 # Dimension of cropped img
MAX_HEIGHT = 256
SAMPLE_NUM = 6  # num of cropped img cropped from raw img
DST_PPC = 256 # pixel per cm
```

4. Generate Label code

```
cd Labelling
python labelling.py
cd ..
```

It encode the labels in original_labels.csv and generate csv files for storing it as explained here https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/

Please rename the csv file to original_labels.csv if needed

You can check labels and corresponding codes in `LabelEncodingMapping.csv` in the subfolder `Labelling`.

Now you are all set :)

## Image Processing

### Run

```
cd ImgProcessing
python main.py
```

The program will run through all images in `RAWIMG_DIR` and store the processed images to subfolders in `PROCESSED_DIR` according to their label code.

It is easier to manage processed images this way, for example easier for counting number of images.

(Optional) Since there are too many classes (85 currently) but not enough data. Currently, labels are named after the texture and color. Thus, we combine the label by color or texture to create two dataset to store in `processed_images_by_color` and `processed_images_by_texture`

By default there are 3 types of combined dataset

Color: Keep the color label only

Texture: Keep the texture label only

Texture2: Keep the general texture label only but ignore -01, -02, etc.

\*\* Reducing number of classes is helpful for machine learning given some classes do not have a lot images. More technical details will be explained in our final report.

## Machine Learning

### Train

1. Define hyperparameters

All hyperparameters are initialized in `Training\paraDict.py`. This is an example of paras dict. Currently, only `ResNet`, `AlexNet` and `VGG` are supported.

```
# ================= Parameters x ======================
PARAS_x = {
    "model_name": "vgg11",
    "model":models.vgg11(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam",
    "scheduler_name":None,
}

PARAS_x['weights_path'] = None
```

Then update the following line to specify the paras `PARAS_x` using in `Training main.py`

```
from paraDict import PARAS_10 as paras
```

2. Run

```
gpu-interactive
conda activate archaeoML
cd Training
python main.py --mode train --by color

```

\*\* There are four options for `by` and the default dataset using is `color`

```
detailed: detailed labels used
color: only color labels used
texture: only texture labels used
texture2: only rough labels used without -01, -02, etc.

```

\*\* The results are logged in `\Training\training_logs`.

3. Update the model path

After training, don't forget to update the model path

```
PARAS_x['weights_path'] = 'vgg11_color_50ep_2023_03_10_12_59'
```

### Test

1. Specify the model used

Update the following line to specify the paras `PARAS_x` using in `Training/main.py`

```
from paraDict import PARAS_10 as paras
```

2. Run

```
python main.py --mode test --by color
```

\*\* The program will load the model in `PARAS_x['weights_path']` to test the dataset specified by `--by`
