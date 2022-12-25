import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import *

env = SnakeEnv(None,15,15)
obs,_ = env.reset()
screen_shape = obs['screen'].shape

x = torch.Tensor(obs['screen']).reshape(1,*screen_shape)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        # a first layer with narrow filters to identify walls, corners, dead ends...
        # returns (w-2)x(h-2) when padding is 'valid'
        self.layer1a = nn.Conv2d(
            in_channels=1,
            out_channels=15, # this determines the number of filters: if one filter, only one output plane...
            kernel_size=3,
            # stride=1,
            padding='valid',
            # dilation=1, #this expands the filter without augmenting the parameters... to identify spatially large patterns with less parameters
            # groups=1, # determines the connection between input channels and output channels
            # bias=True
            # padding_mode='circular' # revert to default which is zeroes... circular breaks it
        )
        
        self.activation1a = nn.ReLU()
        
        # a larger kernel size to identify bigger configurations of smaller patterns?
        # since we are at a 13x13 screen, maybe do a 5x5 kernel
        self.layer2a = nn.Conv2d(
            in_channels=15,
            out_channels=15,
            kernel_size=7,
            stride=4,
            padding='valid'
        )

        self.activation2a = nn.ReLU()

        self.layer3a = nn.Conv2d(
                in_channels=15,
                out_channels=15,
                kernel_size=2,
                padding='valid',
                bias=True
            )
        
        self.activation3a = nn.ReLU()

        # now we join the two features: the result of the various convolutions (which is, one flattened, 15 features)
        # and the one hot encoded direction information (4 neurons)

        self.layer1b = nn.Linear(
            15+4,
            20
        )

        self.activation1b = nn.ReLU()

        # final layer
        self.layer2b= nn.Linear(
            20,
            4
        )

        self.activation2b = nn.Softmax()


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_dict : dict):
        x = self.activation1a(self.layer1a(torch.Tensor(input_dict['screen']).reshape(1,15,15)))
        x = self.activation2a(self.layer2a(x))
        x = self.activation3a(self.layer3a(x))
        y = torch.Tensor(input_dict['direction'])
        z = torch.cat((x.flatten(),y.flatten()))
        z = self.activation1b(self.layer1b(z))
        z = self.activation2b(self.layer2b(z))
        return z

model = DQN()

model(obs)