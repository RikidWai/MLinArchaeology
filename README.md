# MLinArchaeology

### Setup
```
# Setup conda environment
conda create -n archaeoML
conda activate archaeoML
# pip list --format=freeze > requirements.txt

# Setup dependencies
pip install -r requirements.txt
```

Configure your paths and size of cropped images in configure.py

## Image Processing 
### Run
```
cd ImgProcessing
python main.py
```

## Machine Learning 
### Run
```
cd Training
python main.py --mode train
```