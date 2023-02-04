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

# Defines the transformation done to each input data prior to being fed into the model
def create_transform(resize_size=None, crop_size=None):
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  if resize_size and crop_size:
    resize_size = resize_size
    crop_size = crop_size
    # Always ToTensor to be fed into pytorch layers
    transform = transforms.Compose([transforms.Resize(resize_size), 
                                    transforms.CenterCrop(crop_size), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.RandomVerticalFlip(), 
                                    transforms.ToTensor(),transforms.Normalize(
                                        mean=mean,
                                        std=std,
                                    )])
  elif resize_size:
    transform = transforms.Compose([transforms.Resize(resize_size),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=mean,
                                        std=std,
                                    )])
  elif crop_size:
    transform = transforms.Compose([transforms.CenterCrop(crop_size), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.RandomVerticalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=mean,
                                        std=std,
                                    )])
  else:
    transforms.Compose([transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225],
                      )])
  return transform