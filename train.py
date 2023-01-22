# Basic set up
import cv2

import numpy as np
import pandas as pd

import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import os
import time
from pathlib import Path
import matplotlib.pyplot as plt
from Labelling.labelling import getNumClass
# Uncomment if have bugs on GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

datadir = '/content/drive/MyDrive/CS_FYP_archaeology/train_data_dummy/'  # Change to local dir containing training data
NUM_CLASS = getNumClass()  # hard code here

# Shows an image tensor using opencv
# Gives all black? How to show properly without plt?
def imshow_tensor_plt(img_tensor, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    img_tensor = img_tensor.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_tensor = std * img_tensor + mean
        img_tensor = np.clip(img_tensor, 0, 1)

    ax.imshow(img_tensor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


# Defines the transformation done to each input data prior to being fed into the model
def create_transform(resize_size=None, crop_size=None):
    if resize_size and crop_size:
        resize_size = resize_size
        crop_size = crop_size
        # Always ToTensor to be fed into pytorch layers
        transform = transforms.Compose([transforms.Resize(resize_size),
                                        transforms.CenterCrop(crop_size),
                                        transforms.ToTensor()])
    elif resize_size:
        transform = transform.Compose([transforms.Resize(resize_size),
                                       transforms.ToTensor()])
    elif crop_size:
        transform = transform.Compose([transforms.CenterCrop(crop_size),
                                       transforms.ToTensor()])
    else:
        transform = ToTensor()
    return transform


def target_to_oh(target):
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot


if __name__ == '__main__':
    # Loading dataset using default Pytorch ImageFolder
    # Assumes the data structure shown above classified by label into subfolders
    ds = torchvision.datasets.ImageFolder(root=datadir, transform=create_transform(255, 224),
                                          target_transform=target_to_oh)

    # Inspect the classes
    list_of_classes = list(ds.classes)
    print(f'The list of classes: {list_of_classes}')

    dataloader = torch.utils.data.DataLoader(ds, batch_size=2) # Can specify batch_size=1 and shuffle=False

    # Get one batch
    images, labels = next(iter(dataloader))
    imshow_tensor_plt(images[0], normalize=False)
    imshow_tensor_plt(images[1], normalize=False)