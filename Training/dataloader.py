# Basic set up
import sys
sys.path.append('../')

import cv2
import numpy as np
import pandas as pd
import configure as cfg
import pathlib as Path

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import time
import copy

# Custom dataset inheriting the Pytorch generic Dataset
# Use this for higher flexibility, otherwise use ImageFolder for convenience
# Can modify the __getitem__ to customize the data structure returned from each sample
# Works for single folder containing data of all classes, uses csv_file to retrieve label for each image
class SherdDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
      """
      Args:
          csv_file (string): Path to the csv file with (img_path, label) for each row.
          root_dir (string): Directory with all the images.
          transform (callable, optional): Optional transform to be applied
              on a sample.
      """
      self.sherds_frame = pd.read_csv(csv_file)
      self.root_dir = root_dir
      self.transform = transform

    def __len__(self):
      return len(self.sherds_frame)

    def __getitem__(self, idx):
      if torch.is_tensor(idx):
          idx = idx.tolist()

      img_name = os.path.join(self.root_dir, self.sherds_frame.iloc[idx, 0])
      sherd_img = cv2.imread(img_name)
      sherd_label = self.sherds_frame.iloc[idx, 1]

      if self.transform:
          sample = self.transform(sherd_img)

      sample = {'image': sherd_img, 'label': sherd_label}

      return sample