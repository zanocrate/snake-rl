import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import *

env = SnakeEnv(None,10,10)
obs,_ = env.reset()
screen_shape = obs['screen'].shape

x = torch.Tensor(obs['screen']).reshape(-1,*screen_shape)

layer1a = nn.Conv2d(
            in_channels=1,
            out_channels=10, # this determines the number of filters: if one filter, only one output plane...
            kernel_size=3,
            # stride=1,
            padding='same',
            # dilation=1, #this expands the filter without augmenting the parameters... to identify spatially large patterns with less parameters
            # groups=1, # determines the connection between input channels and output channels
            bias=True,
            # padding_mode='circular' # since the snake env is periodic!
        )

layer2a = nn.Conv2d(
            in_channels=10,
            out_channels=10,
            kernel_size=(3*2,3*2),
            stride=(3*2,3*2),
            padding='valid',
            dilation=1,
            bias=True
        )

y=layer1a(x)