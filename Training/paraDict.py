
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
PARAS_1['weights_path'] = 'vgg11_color_50ep_2023_03_15_11_34'
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


PARAS_2['weights_path'] = 'vgg11_color_50ep_2023_03_24_11_15' # Freeze the lower feature layers 


PARAS_2['weights_path'] = 'vgg11_color_50ep_2023_04_06_12_13' # No Augmentation 
PARAS_2['weights_path'] = 'vgg11_texture2_50ep_2023_04_06_16_23' # No Augmentation 

PARAS_2['weights_path'] = 'vgg11_texture2_50ep_2023_04_03_12_53' 
PARAS_2['weights_path'] = 'vgg11_color_50ep_2023_03_14_18_40'

PARAS_2['weights_path'] = 'vgg11_texture_50ep_2023_03_27_17_21' # Freeze first two classifier layers 
PARAS_2['weights_path'] = 'vgg11_color_50ep_2023_03_24_13_38' # Freeze first two classifier layers 
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
PARAS_3['weights_path'] = 'vgg11_bn_color_50ep_2023_03_15_12_29'
# ================= Parameters 4 ====================== 
PARAS_4 = {
    "model_name": "vgg16",
    "model": models.vgg16(weights='DEFAULT'), 
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 1e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD", 
    "scheduler_name":"StepLR",
}
PARAS_4['weights_path'] = 'vgg16_color_50ep_2023_03_09_12_31'
PARAS_4['weights_path'] = 'vgg16_color_50ep_2023_03_16_15_04'
PARAS_4['weights_path'] = 'vgg16_texture2_50ep_2023_04_03_15_32'

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
PARAS_5['weights_path'] = 'vgg19_texture2_50ep_2023_04_04_13_19'
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

PARAS_6['weights_path'] = 'vgg11_texture2_50ep_2023_04_04_14_19'
PARAS_6['weights_path'] = 'vgg11_color_50ep_2023_03_18_11_33'
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
PARAS_8['weights_path'] = 'vgg11_texture2_50ep_2023_04_06_16_23'

# ================= Parameters 9 ====================== 
PARAS_9 = {
    "model_name": "SimNet1",
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-6,
    "num_of_epochs": 10,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam", 
    "scheduler_name": "StepLR",
}

# True creation happens in mlUtils, model here is just for model name
PARAS_9["model"] = cm.SimNet1(conv_out_1=6, conv_out_2=16, hid_dim_1=120, hid_dim_2=60, num_classes=13, kernel_size=5)
# PARAS_9["model"] = cm.SimNet1(conv_out_1=6, conv_out_2=32, hid_dim_1=120, hid_dim_2=60, num_classes=13, kernel_size=5)
# PARAS_9['model'] = None


# ----- Deprecated Start -----
# PARAS_9["optimizer"] = optim.SGD(PARAS_9["model"].parameters(), 
#                                  lr=PARAS_9["learning_rate"], 
#                                  momentum=0.9)
# PARAS_9["exp_lr_scheduler"] = None
# ----- Deprecated End -----

# To specify the path for testing dataset
PARAS_9["weights_path"] = None
# PARAS_9['weights_path'] = 'SimNet1_color_50ep_2023_03_13_00_10'



# ================= Parameters 10 ====================== 

PARAS_10 = {
    "model_name": "alexnet",
    "model": models.alexnet(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam", 
    "scheduler_name": "StepLR",
    "model_architecture": """
	freezed feature 0 to 5
	Adam optimizer
    """
}

PARAS_10["weights_path"] = None
# ================= Parameters 11 ====================== 

PARAS_11 = {
    "model_name": "alexnet",
    "model": models.alexnet(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam", 
    "scheduler_name": "StepLR",
}

PARAS_11['weights_path'] = "alexnet_color_50ep_2023_03_17_16_22_randomRotation_Adam_freezeallExceptLast"
PARAS_11['weights_path'] = 'alexnet_texture2_50ep_2023_04_12_17_25'
# PARAS_11['weights_path'] = None



# ================= Parameters 12 ====================== 
PARAS_12 = {
    "model_name": "resnet34",
    "model": models.resnet34(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-2,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD",
    "scheduler_name": "StepLR"
}
PARAS_12['weights_path'] = None
# PARAS_12['weights_path'] = 'resnet34_texture2_50ep_2023_04_06_01_56'
# PARAS_12['weights_path'] = 'resnet34_texture_50ep_2023_04_07_00_16'
# PARAS_12['weights_path'] = 'resnet34_color_50ep_2023_04_06_00_51'
# PARAS_12['weights_path'] = 'resnet34_color_50ep_freeze3_2023_04_08_18_18'
# PARAS_12['weights_path'] = 'resnet34_texture2_50ep_freezee3_2023_04_08_19_02'
# PARAS_12['weights_path'] = 'resnet34_color_50ep_freeze2_2023_04_08_20_00'
PARAS_12['weights_path'] = 'resnet34_texture2_50ep_freeze2_2023_04_08_20_47'

# ================= Parameters 13 ====================== 
PARAS_13 = {
    "model_name": "resnet50",
    "model": models.resnet50(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-2,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD",
    "scheduler_name": "StepLR"
}
PARAS_13['weights_path'] = None
PARAS_13['weights_path'] = 'resnet50_texture2_50ep_2023_04_06_19_09'
# PARAS_13['weights_path'] = 'resnet50_texture_50ep_2023_04_06_22_34'
# PARAS_13['weights_path'] = 'resnet50_color_50ep_2023_04_06_20_08'




# ================= Parameters 14 ====================== 
PARAS_14 = {
    "model_name": "resnet18",
    "model": models.resnet18(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-2,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD",
    "scheduler_name": "StepLR"
}
PARAS_14['weights_path'] = None
PARAS_14['weights_path'] = 'resnet18_color_50ep_2023_04_08_23_31'


# ================= Parameters 15 ====================== 
# Inspect effect of learning rate
PARAS_15 = {
    "model_name": "resnet18",
    "model": models.resnet18(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "SGD",
    "scheduler_name": "StepLR"
}
PARAS_15['weights_path'] = None
PARAS_15['weights_path'] = 'resnet18_color_50ep_2023_04_09_00_11'


# ================= Parameters 16 ====================== 
# Inspect effect of learning rate
PARAS_16 = {
    "model_name": "resnet18",
    "model": models.resnet18(weights='DEFAULT'),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam",
    "scheduler_name": "StepLR"
}
PARAS_16['weights_path'] = None



# ================= Parameters 17 ====================== 
# Inspects the convergence behavior if weights are randomly initialized
# To compare if this problem persists in LPM over simple model
PARAS_17 = {
    "model_name": "resnet18",
    "model": models.resnet18(weights=None),
    "weights": 'DEFAULT',
    "batch_size": 8,
    "learning_rate": 2e-4,
    "num_of_epochs": 50,
    "loss_func": nn.CrossEntropyLoss(),
    "optimizer_name": "Adam",
    "scheduler_name": "StepLR"
}
PARAS_17['weights_path'] = None





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