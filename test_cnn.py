import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np


class DQN(nn.Module):

    def __init__(self, screen_shape,n_directions, n_actions):
        super(DQN, self).__init__()
        self.layer1a = nn.Conv2d(
            in_channels=1,
            out_channels=5, # this determines the number of filters: if one filter, only one output plane...
            kernel_size=5,
            stride=1,
            padding='same',
            dilation=1, #???
            groups=1, # determines the connection between input channels and output channels
            bias=True,
            padding_mode='circular' # since the snake env is periodic!
        )

        self.layer1b = nn.Linear(n_directions, 4)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_dict : dict):
        x = self.layer1a(input_dict['screen'])

        x = F.relu(self.layer2(x))
        return self.layer3(x)

# With square kernels and equal stride
m = nn.Conv2d(
            in_channels=1,
            out_channels=5, # this determines the number of filters: if one filter, only one output plane...
            kernel_size=5,
            stride=1,
            padding='same',
            dilation=1, #???
            groups=1, # determines the connection between input channels and output channels
            bias=True,
            padding_mode='circular' # since the snake env is periodic!
        )


input = torch.randn(1,1,20,20) # 1 batch size, 1 channel, 20x20
# output = m(input)