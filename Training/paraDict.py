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
    "model": models.resnet18(),
    "batch_size": 8,
    "learning_rate": 2e-2,
    "num_of_epochs": 1,
    "loss_func": nn.CrossEntropyLoss(),
}
PARAS_1["optimizer"] = optim.SGD(PARAS_1["model"].parameters(), 
                                 lr=PARAS_1["learning_rate"], 
                                 momentum=0.9)
PARAS_1["exp_lr_scheduler"] = lr_scheduler.StepLR(PARAS_1["optimizer"], 
                                                  step_size=7, 
                                                  gamma=0.1)
