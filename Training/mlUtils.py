# Basic set up
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import configure as cfg
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor

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
    
def save_training_results(histories, by, num_of_classes, batch_size, learning_rate, num_of_epochs, cnn, loss_func, optimizer, exp_lr_scheduler, data_transforms, time_elapsed, best_acc):

    result_dir = Path(__file__).parent / 'training_logs' / datetime.now().strftime('trResults_%Y_%m_%d_%H_%M')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Record image 
    plot_histories(result_dir, histories)
    
    # Record model details 
    with open(str(result_dir / "paras.txt"), "w") as text_file: 
      text_file.write(
          "List of Parameters:\n"
          f"Dataset: {by}\n"
          f"number of classes: {num_of_classes}\n"
          f"batch_size = {batch_size}\n"
          f"learning_rate = {learning_rate}\n"
          f"num_of_epochs = {num_of_epochs}\n"
          f"cnn = {cnn.__class__.__name__}\n"
          f"loss_func = {loss_func}\n"
          f"optimizer = {optimizer.__class__.__name__}\n"
          f"exp_lr_scheduler = {exp_lr_scheduler}\n"
          f"data_transforms = {data_transforms}\n\n"
          "Results:\n"
          f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n"
          f"Best val Acc: {best_acc:4f}\n"
      )
    # Record histories in csv 
    pd.DataFrame(histories).to_csv(result_dir / 'histories.csv', index=True)
