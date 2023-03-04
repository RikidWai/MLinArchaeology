
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

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
