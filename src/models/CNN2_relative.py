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

    def __init__(self,history_length):
        super(DQN, self).__init__()

        # should result in 60 5x5 matrices
        self.conv1 = nn.Conv2d(
            in_channels=history_length, 
            out_channels=60, 
            kernel_size=3, 
            stride=3,
            padding='valid'
        )
        
        # non linear activation
        self.activation1 = nn.ReLU()
        
        

        # final layer
        self.output_layer= nn.Linear(
            60*5*5,
            3
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = torch.Tensor(x)
        x = self.activation1(self.conv1(x))
        x = self.output_layer(x.flatten(start_dim=1))

        return x