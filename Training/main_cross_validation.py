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
from torch.utils.data import Dataset,ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import KFold,StratifiedKFold

from DatasetUtils import dsUtils 
from dataloader import SherdDataSet
import mlUtils
import customModels as cm

from paraDict import PARAS_15 as paras

import os
import time
import copy
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import scikitplot as skplt
import matplotlib.pyplot as plt
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
k_folds = 5
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


        model.train()  # Set model to training mode
        # else:
        #     model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # clear the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            outputs = model(inputs) # logits of shape (N, C) where N is batch size, C is # classes
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels) # if CELoss: outputs=unnormalized logits; labels=class indices vector of shape (N)

            # backward + optimize only if in training phase

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0) # if CELoss: loss is scalar from logSoftmax and NLLLoss
            running_corrects += torch.sum(preds == labels.data).detach().cpu().numpy()
        # Decays learning rate. 
        if scheduler is not None: 
            scheduler.step() 
        else:
            optimizer.step()

        epoch_loss = running_loss / dataset_sizes['train'] #not correct!!!
        epoch_acc = float(running_corrects) / dataset_sizes['train']

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)


        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # load best model weights
    model.load_state_dict(best_model_wts)

    histories = {
        'train_loss' : train_loss_history,
        'train_acc' : train_acc_history,
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
            # skplt.metrics.plot_roc_curve(labels.cpu().data, preds.cpu())
            # plt.savefig('foo.png')
            # for j in range(inputs.size()[0]):
            #     samples_used += 1
            #     row_num = max(num_samples//2, 1)
            #     ax = plt.subplot(row_num, 3, samples_used)
            #     ax.axis('off')
            #     ax.set_title(f'predicted: {class_names[preds[j]]}')

            #     mlUtils.imshow_list(inputs.cpu().data[j], normalize=False)

            #     if samples_used >= num_samples:
            #         model.train(mode=was_training)
            #         print(f'Accuracy on test images: {100 * correct // total} %')
            #         return 100 * correct // total

        model.train(mode=was_training)
    test_metrics = {
        'recall': recall,
        'precision': precision, 
        'accuracy': accuracy,
        'f_score': f_score
    }
    
    print(f'Accuracy on test images: {100 * correct // total} %')
    return test_metrics


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
        # if not processed_data_dir.exists(): 
            # dsUtils.generateDatasetByFeature(processed_data_dir, FLAGS.by) 
            
        # if not splitted_data_dir.exists():
        #     dsUtils.splitDataset(processed_data_dir, splitted_data_dir) 

        # Transforms data
        data_transforms = {
            'train': mlUtils.create_transform(crop_size=128),
            'test': mlUtils.create_transform(crop_size=128)
        }

        # ===================== Creates datasets and loaders =======================
        # Loading dataset using default Pytorch ImageFolder
        # Assumes the data structure shown above classified by label into subfolders
        image_datasets = {x: datasets.ImageFolder(root=splitted_data_dir / x, 
                                                transform=data_transforms[x]) for x in ['train', 'test']}

        # dataset = ConcatDataset([image_datasets['train'], image_datasets['test']])

        image_dataset = datasets.ImageFolder(processed_data_dir,transform=mlUtils.create_transform(crop_size=128))

        # dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        class_names = image_datasets['train'].classes
        num_classes = len(class_names)
        # Define the K-fold Cross Validator
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
        targets = image_dataset.targets
        # Start print
        print('--------------------------------')
        saveFoldWeightsfolderName = ""
        # For fold results
        results = {}
        # K-fold Cross Validation model evaluation
        for fold, (train_ids, test_ids) in enumerate(kfold.split(image_dataset, targets)):
            # Print
            print(f'FOLD {fold}')
            print('--------------------------------')
            train = torch.utils.data.Subset(image_dataset, train_ids)
            test = torch.utils.data.Subset(image_dataset, test_ids)
            dataset_sizes = {"train": len(train), "test": len(test)}
            print(dataset_sizes)
            train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
            test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
            
            # # Sample elements randomly from a given list of ids, no replacement.
            # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            # test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
            
            # # Define data loaders for training and testing data in this fold
            # trainloader = torch.utils.data.DataLoader(
            #                 dataset, 
            #                 batch_size=batch_size, sampler=train_subsampler)
            # testloader = torch.utils.data.DataLoader(
            #                 dataset,
            #                 batch_size=batch_size, sampler=test_subsampler)

            dataloaders = {"train": train_loader, "test": test_loader}
        
            # # ===================== Instantiates simple cnn model =====================
            # cnn = cm.SimNet1(conv_out_1=4, conv_out_2=6, hid_dim_1=120, hid_dim_2=60, num_classes=num_of_classes, kernel_size=5)
            # cnn = cnn.to(device)

            cnn, _ = mlUtils.initialize_model(model, num_classes, False, weights_path, device)
            optimizer = mlUtils.initialize_optimizer(paras)
            exp_lr_scheduler = mlUtils.initialize_scheduler(paras)

            # ===================== Training and logging results =====================
            model_ft_trained, histories, time_elapsed, best_acc = train_model(cnn, dataloaders, loss_func, optimizer, exp_lr_scheduler, num_of_epochs)
            if saveFoldWeightsfolderName != "":
                mlUtils.save_CV_fold_model(model_ft_trained.state_dict(),
                                        histories, 
                                        FLAGS.by, 
                                        num_classes, 
                                        paras,
                                        data_transforms,
                                        time_elapsed, 
                                        best_acc,
                                        fold, saveFoldWeightsfolderName)

            else:
                saveFoldWeightsfolderName = mlUtils.save_CV_fold_model(model_ft_trained.state_dict(),
                                        histories, 
                                        FLAGS.by, 
                                        num_classes, 
                                        paras,
                                        data_transforms,
                                        time_elapsed, 
                                        best_acc,
                                        fold, "")
            test_metrics = test_model(model, class_names, 4, test_loader)
            results[fold] = test_metrics
    
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        recall_sum = 0.0
        precision_sum = 0.0
        acc_sum = 0.0
        f_score_sum = 0.0

        for key, value in results.items():
            print(f'Fold {key}: {value} ')
            recall_sum += value['recall']
            precision_sum += value['precision']
            acc_sum += value['accuracy']
            f_score_sum += value['f_score']
        print(f'Average Acc: {acc_sum/len(results.items())} %')

        # Save the results
        mlUtils.save_CV_training_results(FLAGS.by, 
                                      num_classes, 
                                      paras,
                                      data_transforms,
                                      saveFoldWeightsfolderName,
                                      results,
                                      k_folds,
                                      recall_sum/len(results.items()),
                                      precision_sum/len(results.items()),
                                      acc_sum/len(results.items()),
                                      f_score_sum/len(results.items()),
                                      )

        print('Finished training!')

    elif Mode == 'test':
        print('\nStart Testing')

        testset = datasets.ImageFolder(root=splitted_data_dir / 'test', 
                                       transform=mlUtils.create_transform(crop_size=128))
        testloader = torch.utils.data.DataLoader(testset, 
                                                 batch_size=5000,
                                                 shuffle=False)
        if paras['weights_path'] is not None:
            weights_path = cfg.MAIN_DIR / 'Training/training_logs'/ paras['weights_path'] / 'weights.pth' 
            class_names = testset.classes
            num_classes = len(class_names)
            model, _ = mlUtils.initialize_model(model, num_classes, False, weights_path, device)
            
            test_metrics = test_model(model, class_names, 4, testloader)
            mlUtils.save_testing_results(weights_path.parent, test_metrics)
            
        print('Finished testing!')
        print()
    else: 
        print('Abort')