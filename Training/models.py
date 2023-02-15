# Self defined neural networks

# To understand more about dimensions through CNNs
# https://stackoverflow.com/questions/66849867/understanding-input-and-output-size-for-conv2d

import torch.nn as nn
import torch.nn.functional as F

# A simple CNN with 2 convolutions and 3 linear layers (2:3)
class SimNet1(nn.Module):

    # One instantiation of the class, customize deeper layers by creating other instantiations of more parameters
    def __init__(self, conv_out_1, conv_out_2, hid_dim_1, hid_dim_2, num_classes, kernel_size):
        super().__init__()

        # === Can customize conv and mlp layers for experimentation ===

        # === Start Conv Layers ===
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv_out_1, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_out_1, conv_out_2, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # self.conv1 = ...

        # === End Conv Layers ===




        # === Start MLP Layers ===
        self.mlp1 = nn.Sequential(
            nn.LazyLinear(hid_dim_1), # if using nn.Linear(), in_dim determined by final conv_out * (image dim after conv)^2
            nn.ReLU(),
            nn.Linear(hid_dim_1, hid_dim_2),
            nn.ReLU(),
            nn.Linear(hid_dim_2, num_classes) # final output dimension matches num of classes
        )

        # self.mlp2 = ...

        # === End MLP Layers ===


    def forward(self, x):
        # Convolutions to abstract information
        x = self.conv1(x)

        # Flatten tensor except batch
        x = torch.flatten(x, 1)

        # Multilayer perceptron layers for learning stage
        x = self.mlp1(x)

        return x





