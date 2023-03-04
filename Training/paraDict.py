# Basic set up
import sys
sys.path.append('../')

import cv2
import numpy as np
import pandas as pd
import configure as cfg


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor

# ================= Parameters 1 ====================== 
PARAS_1 = {
  "model": ,
  "batch_size": "Ford",
  "learning_rate": "Mustang",
  "num_of_epochs": 1964, 
  "loss_func": ,
  "optimizer": , 
  "exp_lr_scheduler": ,
}