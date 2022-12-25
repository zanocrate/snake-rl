import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import * # also imports numpy and pygame

# initialize SnakeEnv with 15x15 grid, no PBC

env = SnakeEnv(
    render_mode = None,
    width=15,
    height=15,
    periodic=False,
    food_reward=1,
    terminated_penalty=-1
)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()

        # a first layer with narrow filters to identify walls, corners, dead ends...
        # returns (w-2)x(h-2) when padding is 'valid'
        self.cnn_layer1 = nn.Conv2d(
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
        
        self.cnn_activation1 = nn.ReLU()
        
        # a larger kernel size to identify bigger configurations of smaller patterns?
        # since we are at a 13x13 screen, maybe do a 5x5 kernel
        self.cnn_layer2 = nn.Conv2d(
            in_channels=15,
            out_channels=15,
            kernel_size=7,
            stride=4,
            padding='valid'
        )

        self.cnn_activation2 = nn.ReLU()

        self.cnn_layer3 = nn.Conv2d(
                in_channels=15,
                out_channels=15,
                kernel_size=2,
                padding='valid',
                bias=True
            )
        
        self.cnn_activation3 = nn.ReLU()

        # now we join the two features: the result of the various convolutions (which is, one flattened, 15 features)
        # and the one hot encoded direction information (4 neurons)

        self.fnn_layer1 = nn.Linear(
            15+4,
            20
        )

        self.fnn_activation1 = nn.ReLU()

        # final layer
        self.fnn_layer2= nn.Linear(
            20,
            4
        )

        self.fnn_activation2 = nn.Softmax()


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_dict : dict):
        x = self.cnn_activation1(self.cnn_layer1(torch.Tensor(input_dict['screen']).reshape(-1,1,15,15)))
        x = self.cnn_activation2(self.cnn_layer2(x))
        x = self.cnn_activation3(self.cnn_layer3(x))
        y = torch.Tensor(input_dict['direction'])
        z = torch.cat((x.reshape(-1,15),y.reshape(-1,4)),dim=1)
        z = self.fnn_activation1(self.fnn_layer1(z))
        z = self.fnn_activation2(self.fnn_layer2(z))
        return z


import random
screen_arr = []
dir_arr = []

env.reset()

for i in range(5):
    obs,rew,term,info=env.step(random.randint(0,3))
    screen_arr.append(obs['screen'])
    dir_arr.append(obs['direction'])

# should have both first dimension as batch size
dir_arr = torch.Tensor(np.array(dir_arr))
screen_arr = torch.Tensor(np.array(screen_arr))

obs_t = {'screen':screen_arr,'direction':dir_arr}

model = DQN()
