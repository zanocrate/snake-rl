# Second CNN model for SnakeEnv with
# width 15
# height 15
# history_length variable but expected 2

import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Class for the Neural Network of the Q value function.
    """

    action_space_type='relative'

    def __init__(self,input_shape):
        
        super(DQN, self).__init__()

        history_length, width, height = input_shape

        if (width % 2 != 0) or (height % 2 != 0):
            raise ValueError('Width and height of input board must be divisible by two.')

        # the number of output features from the convolutions
        out_features = int(     (width-4)*(height-4)        )

        
        # should result in 32 (width-2)x(height-2) planes

        self.conv1 = nn.Conv2d(
            in_channels=history_length, 
            out_channels=32, 
            kernel_size=3,
            padding='valid'
        )

        # non linear activation
        self.relu = nn.ReLU()

        # returns 64 [(width-2) - 2]x[(height-2) - 2] planes
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding='valid'
        )
        
        
        # final layer
        self.output_layer= nn.Linear(
            64*out_features,
            3
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = torch.Tensor(x)
        # padding walls with one?
        # x = torch.nn.functional.pad(x,(1,1,1,1),value=1)
        # x = self.pool1(self.relu(self.conv1(x)))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.output_layer(x.flatten(start_dim=1))

        return x
