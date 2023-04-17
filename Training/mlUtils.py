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
import torch.optim as optim
from torch.optim import lr_scheduler

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
                                    transforms.RandomRotation(90),
                                    transforms.ToTensor(),transforms.Normalize(
                                        mean=mean,
                                        std=std,
                                    )])
  elif resize_size:
    transform = transforms.Compose([transforms.Resize(resize_size),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(90),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=mean,
                                        std=std,
                                    )])
  elif crop_size:
    transform = transforms.Compose([transforms.CenterCrop(crop_size), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(90),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=mean,
                                        std=std,
                                    )])
  else:
    transform = transforms.Compose([
                        transforms.CenterCrop(128),
                        transforms.RandomHorizontalFlip(), 
                        transforms.RandomVerticalFlip(),
                        # transforms.RandomRotation(90),
                        transforms.ToTensor(),
                        transforms.Normalize(
                          mean=mean,
                          std=std,)
                        ])
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
    
    model_architecture = paras.get('model_architecture', 'standard')
    
    # Record model details 
    with open(str(result_dir / "paras.txt"), "w") as text_file: 
      text_file.write(
          "Path: \n"
          f"\t{filename}\n"
          "List of Parameters:\n"
          f"\tModel Architecture: {model_architecture}\n"
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
    return filename

def save_CV_fold_model(model_weights, histories, by, num_of_classes, paras, data_transforms, time_elapsed, best_acc,fold, saveFoldWeightsfolderName):

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    # filename = f"{paras['model_name']}_{by}_fold-{fold}_{paras['num_of_epochs']}ep_{timestamp}"
    filename = f'weights-fold{fold}.pth'

    if saveFoldWeightsfolderName != "":
        result_dir = saveFoldWeightsfolderName
    else:
        result_dir = Path(__file__).parent / 'training_logs' / f"{paras['model_name']}_{by}_CV_{paras['num_of_epochs']}ep_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model_weights, result_dir / filename)
    return result_dir
    
def save_CV_training_results(by, num_of_classes, paras, data_transforms, result_dir, resultsAll, k_folds, recall,precision,acc,f_score):
    
    model_architecture = paras.get('model_architecture', 'standard')
    
    # Record model details 
    with open(str(result_dir / "paras.txt"), "w") as text_file: 
      text_file.write(
          "Path: \n"
          f"\t{result_dir}\n"
          "List of Parameters:\n"
          f"\tModel Architecture: {model_architecture}\n"
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
          f"\tk folds = {k_folds}\n\n"

          "Results:\n"
          f"\tAll results: {resultsAll}\n\n"  
          f"\tAvg. recall: {recall:4f}\n"  
          f"\tAvg. precision: {precision:4f}\n"          
          f"\tAvg. accuracy: {acc:4f}\n"          
          f"\tAvg. f_score: {f_score:4f}\n"          
      )


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
        # Freezing the first few layers from top down
        # Including conv1, bn1, relu, maxpool as first four
        first_n = 4 + 3 # plus: number of layers on top of the first four feature extractors
        ct = 0
        for child in model_ft.children():
            if ct < first_n:
                for param in child.parameters():
                    param.requires_grad = False
            ct += 1

        # Modifies last fully connected layer to match number of classes as output
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
         # Freeze some feature layers (features.0 through features.5)
        for param in model_ft.features[0:6].parameters():
            param.requires_grad = False

        # # Freeze first two classifier layers 
        # for param in model_ft.classifier[0:2].parameters():
        #     param.requires_grad = False
    
        # #Freeze all except the last layer
        # for param in model_ft.parameters():
        #     param.requires_grad = False

        # # Modify the number of channels in the convolutional layers
        # model_ft.features[0] = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        # model_ft.features[3] = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        # model_ft.features[6] = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        # model_ft.features[8] = nn.Conv2d(192, 256, kernel_size=3, padding=1)
        # model_ft.features[10] = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # # Freeze the weights of the pre-trained layers
        # for param in model_ft.features.parameters():
        #     param.requires_grad = False

        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        # print(model_ft.eval())

    elif model_name == "vgg":
        # Freeze the lower feature layers (features.0 through features.2)
        # for param in model_ft.features[0:3].parameters():
        #     param.requires_grad = False

        # # Freeze first two classifier layers 
        # for param in model_ft.classifier[0:2].parameters():
        #     param.requires_grad = False
    
        # #Freeze all except the last layer
        # for param in model_ft.parameters():
        #     param.requires_grad = False
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


def initialize_optimizer(paras):
    optimizer_name = paras["optimizer_name"]
    model = paras["model"]
    lr = paras["learning_rate"]

    print(f'The learning rate is: {lr}')
    
    if optimizer_name == "SGD":
        paras['optimizer'] = optim.SGD(model.parameters(), 
                                        lr=lr, weight_decay=0.0001, 
                                        momentum=paras.get("momentum",0.9))
    elif optimizer_name == "Adam":
        paras['optimizer'] = optim.Adam(model.parameters(), 
                                        lr=lr, 
                                        weight_decay=paras.get('weight_decay',1e-2))
    else:
        print("Invalid optimizer name, exiting...")
        exit()

    return paras['optimizer']

def initialize_scheduler(paras):
    scheduler_name = paras['scheduler_name']
    optimizer = paras['optimizer']
    
    if scheduler_name == None:
        paras['exp_lr_scheduler'] = None 
    elif scheduler_name == "StepLR":
        paras['exp_lr_scheduler'] = lr_scheduler.StepLR(optimizer, 
                                                  step_size=paras.get('step_size', 7), 
                                                  gamma=paras.get('gama', 0.1))
    else:
        print("Invalid scheduler name, exiting...")
        exit()

    return paras['exp_lr_scheduler']

def save_testing_results(dir, test_metrics):
    print(
            f"Best test Recall: {test_metrics['recall']:4f}\n"
            f"Best test Precision: {test_metrics['precision']:4f}\n"
            f"Best test Acc: {test_metrics['accuracy']:4f}\n"
            f"Best test F_score: {test_metrics['f_score']:4f}\n"
    )
    
    with open(dir / "paras.txt", "a") as text_file:
        text_file.write(
            f"\n\tBest test Recall: {test_metrics['recall']:4f}\n"
            f"\tBest test Precision: {test_metrics['precision']:4f}\n"
            f"\tBest test Acc: {test_metrics['accuracy']:4f}\n"
            f"\tBest test F_score: {test_metrics['f_score']:4f}\n"
            f"\tf_score_vec: {test_metrics['f_score_vec']}\n"
            f"\trecall_vec: {test_metrics['recall_vec']}\n"
            f"\precision_vec: {test_metrics['precision_vec']}\n"
        )
