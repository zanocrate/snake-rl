#  CNN model for SnakeEnv with
# even sized screen
import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Class for the Neural Network of the Q value function.
    """

    action_space_types={
        'relative' : 3,
        'absolute' : 4}

    def __init__(self,input_shape,action_space_type):
        
        super(DQN, self).__init__()

        history_length, width, height = input_shape

        assert action_space_type in self.action_space_types.keys()

        if (width % 2 != 0) or (height % 2 != 0):
            raise ValueError('Width and height of input board must be divisible by two.')

        # the number of output features from the convolutions
        out_features = int(     (width-3)*(height-3)        )

        
        # should result in 32 (width-1)x(height-1) planes

        self.conv1 = nn.Conv2d(
            in_channels=history_length, 
            out_channels=32, 
            kernel_size=2,
            padding='valid'
        )

        # non linear activation
        self.relu = nn.ReLU()

        # returns 64 [(width-1) - 2]x[(height-1) - 2] planes
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding='valid'
        )
        
        
        # fc layer
        self.fc= nn.Linear(
            64*out_features,
            20
        )

        self.output_layer = nn.Linear(
            20,
            self.action_space_types[action_space_type]
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = torch.Tensor(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.fc(x.flatten(start_dim=1)))
        x = self.output_layer(x)

        return x