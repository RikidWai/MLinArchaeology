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

# from DatasetUtils import dsUtils 
from dataloader import SherdDataSet
import mlUtils
import customModels as cm

from paraDict import PARAS_15 as paras

# For ensemble use
from paraDict import PARAS_8 as paras_8
from paraDict import PARAS_9 as paras_9
from paraDict import PARAS_11 as paras_11
from paraDict import PARAS_13 as paras_13
from paraDict import PARAS_12 as paras_12
from paraDict import PARAS_14 as paras_14
from paraDict import PARAS_15 as paras_15


import os
import time
import copy
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import scikitplot as skplt
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
import os.path
from os import path
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

print(f'The batch size is {batch_size}')

# ================= Instantiating model globally ======================
model = paras['model']

# print(model)

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

                # Self test print gradients
                # print('Showing conv1 mid conv2d gradients')
                # if model.conv1[3].weight.grad is not None:
                #     print(model.conv1[3].weight.grad[0][0])
                # print('Showing mlp1 mid linear gradients')
                # if model.mlp1[2].weight.grad is not None:
                #     print(model.mlp1[2].weight.grad[0][0])

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

            f_score = f1_score(labels.cpu().data, preds.cpu(), average='macro')
            accuracy = accuracy_score(labels.cpu().data, preds.cpu())
            recall = recall_score(labels.cpu().data, preds.cpu(), average='macro')
            precision = precision_score(labels.cpu().data, preds.cpu(), average='macro')
            
            f_score_vec = f1_score(labels.cpu().data, preds.cpu(), average=None)
            recall_vec = recall_score(labels.cpu().data, preds.cpu(), average=None)
            precision_vec = precision_score(labels.cpu().data, preds.cpu(), average=None)


        model.train(mode=was_training)

    test_metrics = {
        'recall': recall,
        'precision': precision, 
        'accuracy': accuracy,
        'f_score': f_score,
        'f_score_vec': f_score_vec,
        'recall_vec': recall_vec,
        'precision_vec': precision_vec
    }

    print(f'The f1 score vector: {f_score_vec}')
    
    print(f'Accuracy on test images: {100 * correct // total} %')
    return test_metrics

# Test prediction with ensemble method, models is a list of instantialized models to be ensembled
def test_model_ensemble(models, testloader):

    correct = 0
    total = 0
  
    was_training = model.training
    model.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):

            print(f'The number of inputs per enum of testloader: {len(inputs)}')

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Calls multiple models and retrieves their logits vectors
            num_models = len(models)
            outputs = None
            for j in range(num_models):
                if j == 0:
                    outputs = models[j](inputs)
                else:
                    outputs += models[j](inputs) 


            _, preds = torch.max(outputs, 1)


            total += labels.size(0)
            correct += (preds == labels).sum().item()

            f_score = f1_score(labels.cpu().data, preds.cpu(), average='macro')
            accuracy = accuracy_score(labels.cpu().data, preds.cpu())
            recall = recall_score(labels.cpu().data, preds.cpu(), average='macro')
            precision = precision_score(labels.cpu().data, preds.cpu(), average='macro')
            
            f_score_vec = f1_score(labels.cpu().data, preds.cpu(), average=None)
            recall_vec = recall_score(labels.cpu().data, preds.cpu(), average=None)
            precision_vec = precision_score(labels.cpu().data, preds.cpu(), average=None)


        model.train(mode=was_training)

    test_metrics = {
        'recall': recall,
        'precision': precision, 
        'accuracy': accuracy,
        'f_score': f_score,
        'f_score_vec': f_score_vec,
        'recall_vec': recall_vec,
        'precision_vec': precision_vec
    }
    
    print(f'Accuracy on test images: {100 * correct // total} %')
    return test_metrics



# Actual Training

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, test_ensemble')
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

        # Self test on aug
        # data_transforms = {
        #     'train': mlUtils.create_transform(),
        #     'val': mlUtils.create_transform()
        # }

        # ===================== Creates datasets and loaders =======================
        # Loading dataset using default Pytorch ImageFolder
        # Assumes the data structure shown above classified by label into subfolders
        image_datasets = {x: datasets.ImageFolder(root=splitted_data_dir / x, 
                                                transform=data_transforms[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes
        SMOTEomek_output_dir = "/userhome/2072/fyp22007/data/splitted_processed_images_by_color/train_SMOTEomek/"

        '''
        X_train = []
        y_train = []
        for i in range(len(image_datasets['train'])):
            temp_x, temp_y = image_datasets['train'][i]
            X_train.append(temp_x)
            y_train.append(temp_y)
        X_train = torch.stack(X_train)
        print(X_train.size())
        X_train = X_train.view(X_train.shape[0], -1)
        print(X_train.size())

        X_train = X_train.numpy()
        y_train = np.asarray(y_train)            

        # Apply SMOTETomek to the training set
        smote_tomek = SMOTETomek(random_state=42)
        x_smote, y_smote = smote_tomek.fit_resample(X_train, y_train)
        x_smote.reshape(x_smote.shape[0], 3,128,128)
        for class_name in class_names:
            if not path.exists(SMOTEomek_output_dir + class_name):
                os.mkdir(SMOTEomek_output_dir + class_name)
        x_tensor, y_tensor = torch.from_numpy(x_smote), torch.from_numpy(y_smote)
       
        from PIL import Image
        from torchvision import transforms
        # Define a transform to convert the tensor to a PIL Image object
        to_pil = transforms.ToPILImage()

        # Save the resampled images to disk and create a new ImageFolder object
        for i in range(x_smote.shape[0]):
            image = x_tensor[i].numpy()
            label = y_tensor[i].item()

            class_name = class_names[label]
            path = SMOTEomek_output_dir + class_name + '/{}.jpg'.format(i)

            # Convert the image array to a PIL Image object
            pil_image = to_pil(image)

            # Convert the pixel values to the range [0, 255] and cast to uint8
            pil_image = pil_image.convert('RGB')
            pil_image = pil_image.point(lambda x: x * 255).convert('L').convert('RGB')
            pil_image = pil_image.convert('RGB')

            # Save the image as a JPEG file
            pil_image.save(path, format='JPEG')

        # # Save the resampled images to disk and create a new ImageFolder object
        # for i in range(x_smote.shape[0]):
        #     image = x_tensor[i].numpy()
        #     label = y_tensor[i].item()
            
        #     class_name = class_names[label]
        #     path = SMOTEomek_output_dir + class_name + '/{}.jpg'.format(i)
            
        #     # Convert the image array to a PIL Image object
        #     pil_image = Image.fromarray(image)
        #     if pil_image.mode != 'RGB':
        #         pil_image = pil_image.convert('RGB')
            
        #     # Save the image as a JPEG file
        #     pil_image.save(path, format='JPEG')
        
        image_datasets['train'] = datasets.ImageFolder(root=SMOTEomek_output_dir, transform=data_transforms['train'])

        '''
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                      batch_size=batch_size,
                                                    shuffle=True)
                                                    for x in ['train', 'val']}

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        num_classes = len(class_names)

        
        
        # # ===================== Instantiates simple cnn model =====================
        # cnn = cm.SimNet1(conv_out_1=4, conv_out_2=6, hid_dim_1=120, hid_dim_2=60, num_classes=num_of_classes, kernel_size=5)
        # cnn = cnn.to(device)

        cnn, _ = mlUtils.initialize_model(model, num_classes, False, weights_path, device)
        optimizer = mlUtils.initialize_optimizer(paras)
        exp_lr_scheduler = mlUtils.initialize_scheduler(paras)

        # ===================== Training and logging results =====================
        # Assumes cnn is properly initialized first, same structure as the one initialized before training
        model_ft_trained, histories, time_elapsed, best_acc = train_model(cnn, dataloaders, loss_func, optimizer, exp_lr_scheduler, num_of_epochs)


        # Save the results
        paras['weights_path'] = mlUtils.save_training_results(model_ft_trained.state_dict(),
                                      histories, 
                                      FLAGS.by, 
                                      num_classes, 
                                      paras,
                                      data_transforms,
                                      time_elapsed, 
                                      best_acc)

        print('Finished training! continue testing')
        Mode = 'test'

    if Mode == 'test':
        print('\nStart Testing')

        testset = datasets.ImageFolder(root=splitted_data_dir / 'test', 
                                       transform=mlUtils.create_transform(crop_size=128))
        testloader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=5000,
                                                 shuffle=False)
        print("weight path:",paras['weights_path'])
        if paras['weights_path'] is not None:
            weights_path = cfg.MAIN_DIR / 'Training/training_logs'/ paras['weights_path'] / 'weights.pth' 
            class_names = testset.classes
            num_classes = len(class_names)
            model, _ = mlUtils.initialize_model(model, num_classes, False, weights_path, device)
            
            test_metrics = test_model(model, class_names, 4, testloader)
            mlUtils.save_testing_results(weights_path.parent, test_metrics)
            
        print('Finished testing!')
        print()
    
    if Mode == 'test_ensemble':
        print('\nStart Testing Ensemble Method')

        testset = datasets.ImageFolder(root=splitted_data_dir / 'test', 
                                       transform=mlUtils.create_transform(crop_size=128))
        testloader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=5000,
                                                 shuffle=False)

        # Holds the set of models used for ensemble prediction
        models = []
        # Modify the preferred selection of models using PARAS
        paras_ensembled = [paras_15, paras_9] # [paras_13, paras_12, paras_14, paras_11, paras] # [paras_8, paras_11, paras_13, paras_12, paras_14]
        
        for i in range(len(paras_ensembled)):
            if paras_ensembled[i]['weights_path'] is not None:
                weights_path = cfg.MAIN_DIR / 'Training/training_logs'/ paras_ensembled[i]['weights_path'] / 'weights.pth' 
                class_names = testset.classes
                num_classes = len(class_names)
                model, _ = mlUtils.initialize_model(paras_ensembled[i]['model'], num_classes, False, weights_path, device)
                models.append(model)

        test_metrics = test_model_ensemble(models, testloader)
        print(test_metrics)
                
        print('Finished testing ensemble!')
        print()



    else: 
        print('Abort')