from env import *
import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    """
    Class for the Neural Network of the Q value function.
    """
    def __init__(self,history_length):
        super(DQN, self).__init__()

        # a first layer with narrow filters to identify walls, corners, dead ends...
        # returns (w-2)x(h-2) when padding is 'valid'
        self.conv1 = nn.Conv2d(
            in_channels=history_length, # the number of states fed
            out_channels=15, # the number of kernels
            kernel_size=3, # a single int means square kernel
            stride=1, # it's the default: no skipping
            padding='valid', # no padding
            # dilation=1, #this expands the filter without augmenting the parameters... to identify spatially large patterns with less parameters
            # groups=1, # determines the connection between input channels and output channels
            # bias=True
            # padding_mode='circular' # revert to default which is zeroes... circular breaks it
        )
        
        # non linear activation
        self.activation1 = nn.ReLU()
        
        # a larger kernel size to identify bigger configurations of smaller patterns?
        # since we are at a 13x13 screen, maybe do a 5x5 kernel
        self.conv2 = nn.Conv2d(
            in_channels=15,
            out_channels=15,
            kernel_size=7,
            stride=4,
            padding='valid'
        )

        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
                in_channels=15,
                out_channels=15,
                kernel_size=2,
                padding='valid',
                bias=True
            )
        
        self.activation3 = nn.ReLU()


        # now the feed forward layers
        self.ffl1 = nn.Linear(
            15,
            256
        )

        self.ffl_activation1 = nn.ReLU()

        # final layer
        self.ffl2= nn.Linear(
            256,
            4
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = torch.Tensor(x)
        x = self.activation1(self.conv1(x))
        x = self.activation2(self.conv2(x))
        x = self.activation3(self.conv3(x))
        x = self.ffl_activation1(self.ffl1(x.flatten(start_dim=1))) # do not drop the batch dimension
        x = self.ffl2(x)

        return x

# game parameters

WIDTH = 15
HEIGHT = 15
HISTORY_LENGTH = 4

env = SnakeEnv(
    render_mode='human',
    width=WIDTH,
    height=HEIGHT,
    periodic=False,
    observation_type=1, # this returns only the screen
    history_length=HISTORY_LENGTH
)
env.metadata["render_fps"] = 10 # default is 4
action_to_string = {
    0:"UP",
    1:"DOWN",
    2:"RIGHT",
    3:"LEFT"
}

terminated = False

observation, info = env.reset()

policy_net = torch.load('./checkpoint/policy_net.20.pth')

while not terminated:
    
    observation = np.expand_dims(observation,0)

    with torch.no_grad():
        actions = policy_net(observation)
        action = int(actions.argmax(-1))

    observation,reward,terminated,info = env.step(action)
    
