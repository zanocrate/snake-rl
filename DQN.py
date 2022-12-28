import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Class for the Neural Network of the Q value function.
    Takes as input a dictionary:
    input['screen'] = 15x15 matrix representing the screen
    input['direction'] = one hot encoded vector representation of the direction
    """
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

        self.fnn_activation2 = nn.Softmax(dim=0)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, input_dict : dict):

        x = self.cnn_activation1(self.cnn_layer1(torch.Tensor(input_dict['screen']).reshape(-1,1,15,15)))
        x = self.cnn_activation2(self.cnn_layer2(x))
        x = self.cnn_activation3(self.cnn_layer3(x))
        y = torch.Tensor(input_dict['direction'])
        z = torch.cat((x[:,:,0,0],y.reshape(-1,4)),dim=1)
        z = self.fnn_activation1(self.fnn_layer1(z))
        # z = self.fnn_activation2(self.fnn_layer2(z))
        z = self.fnn_layer2(z)
        return z

# visualize model

if __name__ == '__main__':

    import torchviz
    from env import *

    env = SnakeEnv(None,15,15)
    obs,_ = env.reset()

    model = DQN()

    y = model(obs)

    torchviz.make_dot(y,params=dict(model.named_parameters())).render("./img/dqn_torchviz", format="png")