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
import customModels as cm

from paraDict import PARAS_10 as paras

import os
import time
import copy
from datetime import datetime

from pathlib import Path
# Uncomment if have bugs on GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)

if str(device)=='cpu':
    print('Quitting due to using cpu')
    # sys.exit(0)


# paras = paraDict.PARAS_1
# ================= Hyperparameters ====================== 
batch_size = paras['batch_size']
learning_rate = paras['learning_rate']
num_of_epochs = paras['num_of_epochs']

# ================= Instantiating model globally ======================
# cnn = models.resnet18(weights='DEFAULT')
model = paras['model']

# ================= Loss, optimizer and scheduler ======================
loss_func = paras['loss_func']
# optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# optimizer = paras['optimizer']
# exp_lr_scheduler =  paras['exp_lr_scheduler'] # Decay LR by a factor of 0.1 every 7 epochs

# ================= Helper functions for training and testing ======================
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # Modes determine activation of dropout layers
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # clear the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # logits of shape (N, C) where N is batch size, C is # classes
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels) # if CELoss: outputs=unnormalized logits; labels=class indices vector of shape (N)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) # if CELoss: loss is scalar from logSoftmax and NLLLoss
                running_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()
            if phase == 'train': 
                # Decays learning rate. 
                if scheduler is not None: 
                    scheduler.step() 
                else:
                    optimizer.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            else:
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)


        print()

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

def test_model(model, class_names, num_samples, testloader):

    correct = 0
    total = 0
    samples_used = 0
  
    was_training = model.training
    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for j in range(inputs.size()[0]):
                samples_used += 1
                row_num = max(num_samples//2, 1)
                ax = plt.subplot(row_num, 3, samples_used)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')

                mlUtils.imshow_list(inputs.cpu().data[j], normalize=False)

                if samples_used >= num_samples:
                    model.train(mode=was_training)
                    print(f'Accuracy on test images: {100 * correct // total} %')
                    return 100 * correct // total

        model.train(mode=was_training)

    return 100 * correct // total


# Actual Training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='options: train, test')
    parser.add_argument('--by', type=str, default='color', help='options: detailed, color, texture, texture2')
    FLAGS = parser.parse_args()
    
    Mode = FLAGS.mode
    dir = 'processed_images' + ('' if FLAGS.by == 'detailed' else f'_by_{FLAGS.by}')
    print(dir)
    processed_data_dir = cfg.DATA_DIR / dir
    splitted_data_dir = cfg.DATA_DIR / ('splitted_' + dir)
    # dsUtils.splitDataset() # Uncomment this line if needed 
    weights_path = None
    
    if Mode == 'train':
        print('\nStart Training')

        # ===================== Retrieves usable data =======================
        # Splits and selects data
        if not processed_data_dir.exists(): 
            dsUtils.generateDatasetByFeature(processed_data_dir, FLAGS.by) 
            
        if not splitted_data_dir.exists():
            dsUtils.splitDataset(processed_data_dir, splitted_data_dir) 

        # Transforms data
        data_transforms = {
            'train': mlUtils.create_transform(crop_size=128),
            'val': mlUtils.create_transform(crop_size=128)
        }

        # ===================== Creates datasets and loaders =======================
        # Loading dataset using default Pytorch ImageFolder
        # Assumes the data structure shown above classified by label into subfolders
        image_datasets = {x: datasets.ImageFolder(root=splitted_data_dir / x, 
                                                transform=data_transforms[x]) for x in ['train', 'val']}

        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                      batch_size=batch_size,
                                                    shuffle=True)
                                                    for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        
        
        # # ===================== Instantiates simple cnn model =====================
        # cnn = cm.SimNet1(conv_out_1=4, conv_out_2=6, hid_dim_1=120, hid_dim_2=60, num_classes=num_of_classes, kernel_size=5)
        # cnn = cnn.to(device)

        cnn, _ = mlUtils.initialize_model(model, num_classes, False, weights_path, device)
        optimizer = mlUtils.initialize_optimizer(paras)
        exp_lr_scheduler = mlUtils.initialize_scheduler(paras)

        # ===================== Training and logging results =====================
        # model_ft needs to be properly initialized first, same structure as the one initialized before training
        # model_used = 'alexnet'
        # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

        # # Paths for weights saving and loading
        # weights_save_file = f'weights/{model_used}_{FLAGS.by}_{num_of_epochs}ep_{timestamp}.pth'
        # weights_load_file = 'weights/simnet1_color_2ep_2023_03_01_22_07.pth' # Modify this when loading

        # cnn.load_state_dict(torch.load('weights/resNET_model_weights_color.pth'))

        # cnn.load_state_dict(torch.load(weights_load_file)) # -- newly added
        model_ft_trained, histories, time_elapsed, best_acc = train_model(cnn, dataloaders, loss_func, optimizer, exp_lr_scheduler, num_of_epochs)
        # torch.save(model_ft_trained.state_dict(), f'weights/resNET_model_weights_{FLAGS.by}_100epoch.pth')

        # torch.save(model_ft_trained.state_dict(), weights_save_file) # -- newly added

        # Save the results
        mlUtils.save_training_results(model_ft_trained.state_dict(),
                                      histories, 
                                      FLAGS.by, 
                                      num_classes, 
                                      paras,
                                      data_transforms,
                                      time_elapsed, 
                                      best_acc)

        print('Finished training!')

    elif Mode == 'test':
        print('\nStart Testing')

        testset = datasets.ImageFolder(root=splitted_data_dir / 'test', 
                                       transform=mlUtils.create_transform(crop_size=128))
        testloader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=batch_size,
                                                 shuffle=True)
        if paras['weights_path'] is not None:
            weights_path = cfg.MAIN_DIR / 'Training/training_logs'/ paras['weights_path'] / 'weights.pth' 
            class_names = testset.classes
            num_classes = len(class_names)
            model, _ = mlUtils.initialize_model(model, num_classes, False, weights_path, device)
            
            test_accuracy = test_model(model, class_names, 4, testloader)
            
            print(f'Accuracy on test images: {test_accuracy} %')
            mlUtils.save_testing_results(weights_path.parent, test_accuracy)
            
        print('Finished testing!')
        print()
    else: 
        print('Abort')