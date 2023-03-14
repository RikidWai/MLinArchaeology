
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models

import customModels as cm

# ================= Parameters 1 ====================== 
PARAS_1 = {
    "model_name": "vgg11",
    "model": models.vgg11(weights=None),
    "weights": None,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD", 
    "scheduler_name":"StepLR",
}
PARAS_1['weights_path'] = 'vgg11_color_50ep_2023_03_08_11_03'

# ================= Parameters 2 ====================== 
PARAS_2 = {
    "model_name": "vgg11",
    "model": models.vgg11(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD", 
    "scheduler_name":"StepLR",
}
PARAS_2['weights_path'] = 'vgg11_color_50ep_2023_03_08_11_48'

# ================= Parameters 3 ====================== 
PARAS_3 = {
    "model_name": "vgg11_bn",
    'model': models.vgg11_bn(weights='DEFAULT'), 
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD", 
    "scheduler_name":"StepLR",
}
PARAS_3['weights_path'] = 'vgg11_bn_color_50ep_2023_03_09_10_51'

# ================= Parameters 4 ====================== 
PARAS_4 = {
    "model_name": "vgg16",
    "model": models.vgg16(weights='DEFAULT'), 
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD", 
    "scheduler_name":"StepLR",
}
PARAS_4['weights_path'] = 'vgg16_color_50ep_2023_03_09_12_31'

# ================= Parameters 5 ====================== 
PARAS_5 = {
    "model_name": "vgg19",
    "model":models.vgg19(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD", 
    "scheduler_name":"StepLR",
    
}
PARAS_5['weights_path'] = 'vgg19_color_50ep_2023_03_09_15_04'

# ================= Parameters 6 ====================== 
PARAS_6 = {
    "model_name": "vgg11",
    "model": models.vgg11(weights='DEFAULT'), 
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam", 
    "scheduler_name":"StepLR",
}
PARAS_6['weights_path'] = 'vgg11_color_50ep_2023_03_09_17_18'

# ================= Parameters 7 ====================== 
PARAS_7 = {
    "model_name": "vgg11",
    "model": models.vgg11(weights=None),
    "weights": None,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam", 
    "scheduler_name":None,
}
PARAS_7['weights_path'] = None


# ================= Parameters 8 ====================== 
PARAS_8 = {
    "model_name": "vgg11",
    "model":models.vgg11(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam", 
    "scheduler_name":None,
}

PARAS_8['weights_path'] = 'vgg11_color_50ep_2023_03_10_12_59'

# ================= Parameters 9 ====================== 
PARAS_9 = {
    "model_name": "SimNet1",
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
}
PARAS_9["model"] = cm.SimNet1(conv_out_1=6, conv_out_2=16, hid_dim_1=120, hid_dim_2=60, num_classes=13, kernel_size=5)
PARAS_9["optimizer"] = optim.SGD(PARAS_9["model"].parameters(), 
                                 lr=PARAS_9["learning_rate"], 
                                 momentum=0.9)
PARAS_9["exp_lr_scheduler"] = None
PARAS_9['weights_path'] = 'SimNet1_color_50ep_2023_03_13_00_10' # To specify the path for testing dataset

# # ================= Parameters 9 ====================== 
# PARAS_9 = {
#     "model_name": "resnet18",
#     "model": models.resnet18(weights=None),
#     "batch_size": 8,
#     "learning_rate": 2e-2,
#     "num_of_epochs": 1,
#     "loss_func": nn.CrossEntropyLoss(),
# }
# PARAS_9["optimizer"] = optim.SGD(PARAS_9["model"].parameters(), 
#                                  lr=PARAS_9["learning_rate"], 
#                                  momentum=0.9)
# PARAS_9["exp_lr_scheduler"] = lr_scheduler.StepLR(PARAS_9["optimizer"], 
#                                                   step_size=7, 
#                                                   gamma=0.1)