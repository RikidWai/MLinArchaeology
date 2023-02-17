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
import matplotlib.pyplot as plt
import argparse

from DatasetUtils import dsUtils 
from dataloader import SherdDataSet
import mlUtils 

# import models

import os
import time
import copy

from pathlib import Path
# Uncomment if have bugs on GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

# Parameters 
batch_size = 8
learning_rate = 2e-4
num_of_epochs = 1

# cnn = models.resnet18(weights='DEFAULT')
cnn = models.resnet18()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # Decay LR by a factor of 0.1 every 7 epochs

# TODO: Add dataloaders as arg
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # for epoch in range(num_epochs):
    #     print(f'Epoch {epoch}/{num_epochs - 1}')
    #     print('-' * 10)

    #     # Each epoch has a training and validation phase
    #     for phase in ['train', 'val']:
    #         # Modes determine activation of dropout layers
    #         if phase == 'train':
    #             model.train()  # Set model to training mode
    #         else:
    #             model.eval()   # Set model to evaluate mode

    #         running_loss = 0.0
    #         running_corrects = 0

    #         # Iterate over data.
    #         for inputs, labels in dataloaders[phase]:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             # clear the parameter gradients
    #             optimizer.zero_grad()

    #             # forward
    #             # track history if only in train
    #             with torch.set_grad_enabled(phase == 'train'):
    #                 outputs = model(inputs) # logits of shape (N, C) where N is batch size, C is # classes
    #                 _, preds = torch.max(outputs, 1)
    #                 loss = criterion(outputs, labels) # if CELoss: outputs=unnormalized logits; labels=class indices vector of shape (N)

    #                 # backward + optimize only if in training phase
    #                 if phase == 'train':
    #                     loss.backward()
    #                     optimizer.step()

    #             # statistics
    #             running_loss += loss.item() * inputs.size(0) # if CELoss: loss is scalar from logSoftmax and NLLLoss
    #             running_corrects += torch.sum(preds == labels.data)
    #         if phase == 'train':
    #             scheduler.step() # Decays learning rate. If not using scheduler, replace with optimizer.step()

    #         epoch_loss = running_loss / dataset_sizes[phase]
    #         epoch_acc = running_corrects.double() / dataset_sizes[phase]

    #         print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    #         # deep copy the model
    #         if phase == 'val' and epoch_acc > best_acc:
    #             best_acc = epoch_acc
    #             best_model_wts = copy.deepcopy(model.state_dict())
            
    #         if phase == 'val':
    #             val_loss_history.append(epoch_loss)
    #             val_acc_history.append(epoch_acc)
    #         else:
    #             train_loss_history.append(epoch_loss)
    #             train_acc_history.append(epoch_acc)


    #     print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    histories = {
        'train_loss' : train_loss_history,
        'val_loss' : val_loss_history,
        'train_acc' : train_acc_history,
        'val_acc' : val_acc_history,
    }

    return model, histories, time_elapsed, best_acc

def test():
    return

# Actual Training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='options: train, test')
    parser.add_argument('--by', type=str, default='detailed', help='options: detailed, color, texture')
    FLAGS = parser.parse_args()
    
    Mode = FLAGS.mode
    By = '' if FLAGS.by == 'detailed' else FLAGS.by 
    
    datadir = cfg.SPLITTED_DIR /  By
    # dsUtils.splitDataset() # Uncomment this line if needed 
    
    if Mode == 'train':
        # Loading dataset using default Pytorch ImageFolder
        # Assumes the data structure shown above classified by label into subfolders
        #TODO 
        # Split and select data
        
        # ds = torchvision.datasets.ImageFolder(root=datadir / 'train', transform=create_transform(255, 224))

        # # Certain models e.g. Inception v3 requires certain size of images
        # # Skipping normalization here
        # # Assumes data images are all 170x170
        data_transforms = {
            'train': mlUtils.create_transform(crop_size=128),
            'val': mlUtils.create_transform(crop_size=128)
        }
        # Pytorch losses like CELoss do not require one-hot labels

        # image_datasets = {x: datasets.ImageFolder(root=os.path.join(data_dir, x),
        #                   transform=data_transforms[x], target_transform=target_to_oh)
        #                   for x in ['train', 'val']}
        image_datasets = {x: datasets.ImageFolder(root=datadir / x, 
                                                transform=data_transforms[x]) for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                        shuffle=True)
                        for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        num_of_classes = len(class_names)
        cnn.fc = nn.Linear(cnn.fc.in_features, num_of_classes)
        cnn = cnn.to(device)
        # model_ft needs to be properly initialized first, same structure as the one initialized before training
        # cnn.load_state_dict(torch.load('weights/flip_resnet18_model_weights.pth'))
        model_ft_trained, histories, time_elapsed, best_acc = train_model(cnn, dataloaders, loss_func, optimizer, exp_lr_scheduler, num_of_epochs)
        # torch.save(model_ft_trained.state_dict(), 'weights/flip_resnet18_model_weights_100epoch.pth')

        # Save the results


        mlUtils.save_training_results(histories, 
                                      FLAGS.by, 
                                      num_of_classes, 
                                      batch_size, 
                                      learning_rate, 
                                      num_of_epochs, 
                                      cnn, 
                                      loss_func, 
                                      optimizer, 
                                      exp_lr_scheduler, 
                                      data_transforms,
                                      time_elapsed, 
                                      best_acc)


    elif Mode == 'test':
        print(Mode)


    else: 
        print('Abort')