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
    def __init__(self,history_length):
        super(DQN, self).__init__()

        # should result in 30 5x5 matrices
        self.conv1 = nn.Conv2d(
            in_channels=history_length, 
            out_channels=30, 
            kernel_size=3, 
            stride=3,
            padding='valid'
        )
        
        # non linear activation
        self.activation1 = nn.ReLU()
        
        


        # a single feed forward layer
        self.ffl1 = nn.Linear(
            30*5*5,
            256
        )

        self.ffl_activation1 = nn.ReLU()

        # final layer
        self.output_layer= nn.Linear(
            256,
            4
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = torch.Tensor(x)
        x = self.activation1(self.conv1(x))
        x = self.ffl_activation1(self.ffl1(x.flatten(start_dim=1))) # do not drop the batch dimension
        x = self.output_layer(x)

        return x