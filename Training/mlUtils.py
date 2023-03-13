# Basic set up
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import configure as cfg
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
import torch
import customModels as cm

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


# Takes in histories object which is a tuple of length 4
# Shows the variations of lossses and accuracies over epochs
def plot_histories(dir, histories):

    epochs = range(1, len(histories['train_acc']) + 1)

    plt.plot(epochs, histories['train_acc'], 'r', label='Training acc')
    plt.plot(epochs, histories['val_acc'], 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(dir / 'accuracy.png', bbox_inches='tight')
    plt.figure()

    plt.plot(epochs, histories['train_loss'], 'r', label='Training loss')
    plt.plot(epochs, histories['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    plt.savefig(dir / 'loss.png', bbox_inches='tight')
    
def save_training_results(model_weights, histories, by, num_of_classes, paras, data_transforms, time_elapsed, best_acc):

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    filename = f"{paras['model_name']}_{by}_{paras['num_of_epochs']}ep_{timestamp}"
    
    result_dir = Path(__file__).parent / 'training_logs' / filename
    result_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model_weights, result_dir / 'weights.pth')
    
    # Record image 
    plot_histories(result_dir, histories)
    
    # Record model details 
    with open(str(result_dir / "paras.txt"), "w") as text_file: 
      text_file.write(
          "Path: \n"
          f"\t{filename}\n"
          "List of Parameters:\n"
          f"\tDataset: {by}\n"
          f"\tcnn = {paras['model_name']}\n"
          f"\tnumber of classes: {num_of_classes}\n"
          f"\tbatch_size = {paras['batch_size']}\n"
          f"\tlearning_rate = {paras['learning_rate']}\n"
          f"\tweights = {paras['weights']}\n"
          f"\tnum_of_epochs = {paras['num_of_epochs']}\n"
          f"\tloss_func = {paras['loss_func']}\n"
          f"\toptimizer = {paras['optimizer'].__class__.__name__}\n"
          f"\texp_lr_scheduler = {paras['exp_lr_scheduler'].__class__.__name__}\n"
          f"\tdata_transforms = {data_transforms}\n\n"
          "Results:\n"
          f"\tTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n"
          f"\tBest val Acc: {best_acc:4f}\n"          
      )
    # Record histories in csv 
    pd.DataFrame(histories).to_csv(result_dir / 'histories.csv', index=True)

def imshow_list(inp, title=None, normalize=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if normalize:
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      inp = std * inp + mean
      inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def initialize_model(model_ft, num_classes, feature_extract, weights_path = None, device = torch.device('cuda')  ):
    # Initialize these variables which will be set in this if statement. Each of these
    #  variables is model specific.
    
    
    model_name = model_ft.__class__.__name__.lower()
    input_size = 0
    print(f'model_name is {model_name}')
    if model_name == "resnet":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "simnet1":
        # Creation of model already done in paraDict
        print('using SimNet1')

        # model_ft = cm.SimNet1(conv_out_1=4, conv_out_2=6, hid_dim_1=120, hid_dim_2=60, num_classes=num_of_classes, kernel_size=5)
        # model_ft = cm.SimNet1(conv_out_1=6, conv_out_2=16, hid_dim_1=120, hid_dim_2=60, num_classes=num_of_classes, kernel_size=5)
    else:
        print("Invalid model name, exiting...")
        exit()
        
    if weights_path is not None: 
        model_ft.load_state_dict(torch.load(weights_path))
    return model_ft.to(device), input_size

def save_testing_results(dir, accuracy):
    with open(dir / "paras.txt", "a") as text_file:
        text_file.write(
            f'\tBest test Acc: {accuracy:4f}\n'
        )