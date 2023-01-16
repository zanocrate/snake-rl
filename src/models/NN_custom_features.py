# NN of built features
# for relative action space
# any width
# any height
# history_length must be at least 3
#
# built features are:
# food direction from head relative to the direction snake is going
# how many free blocks are in front of the snake head
# same for left and right of the snake head

import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Class for the Neural Network of the Q value function.
    """
    
    rotation_matrix_90ccw = torch.Tensor([[0,-1],[1,0]]).to(int)
    rotation_matrix_90cw = torch.Tensor([[0,1],[-1,0]]).to(int)
    
    def __init__(self,history_length):
        
        # call parent class init
        super(DQN, self).__init__()
        
        assert history_length >= 3, f"history_length >= 3 expected, got: {history_length}"
        
        # to hidden layer of 20 neurons
        self.layer1 = nn.Linear(
            5,
            20
        )
        
        # non linear activation
        self.activation = nn.ReLU()
        
        # final layer
        self.output_layer= nn.Linear(
            20,
            3
        )
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        
        x = torch.Tensor(x)
        x = self._get_features(x)
        x = self.activation(self.layer1(x))
        x = self.output_layer(x)

        return x    
    
    def _get_features(self,x):
            """
            Returns a tensor like (batch_size,n_features) from a (batch_size,history_length,width,height) tensor.
            """
            
            batch_size = x.shape[0]
            
            # we have K>=3 history frames; we want info from the 0th frame (the latest)
            # first we will get the food coordinates
            
            food_coord=torch.argwhere(x[:,0,:,:] == -1)[:,1:] # shaped like (batch_size,2)
            
            # frame == 1 tells us where the snake is; logical or between that of 0 and 1 frames tells us the union
            # intersect the union with the 0 frame using a logical and we grab the head
            head_coord_bool=torch.logical_and(torch.logical_or(x[:,0,:,:]==1 ,x[:,1,:,:]==1),x[:,0,:,:]==1)
            
            # finally grab the coordinates
            head_coord=torch.argwhere(head_coord_bool==1)[:,1:] # shaped like (batch_size,2)
            
            # same but with frames 1 and 2 to grab previous head coord
            prev_head_coord_bool=torch.logical_and(torch.logical_or(x[:,1,:,:]==1 ,x[:,2,:,:]==1),x[:,1,:,:]==1)
            prev_head_coord=torch.argwhere(prev_head_coord_bool==1)[:,1:] # shaped like (batch_size,2)
            
            direction=head_coord - prev_head_coord # shaped like (batch_size,2)
            
            diff = food_coord - head_coord
            
            for b in range(batch_size):
                k=0
                while not torch.eq(torch.linalg.matrix_power(self.rotation_matrix_90ccw,k) @ direction[b],torch.Tensor([1,0])).all():
                    k+=1
                # k is now the number of times we need to rotate the direction of the snake to be [1,0]
                diff[b] = torch.linalg.matrix_power(self.rotation_matrix_90ccw,k) @ diff[b]
            
            # now diff is a tensor of shape (batch_size,2) telling the vector that point towards the food
            # from the head snake, in the reference frame in which his direction is always [1,0]
            
            # initialize a tensor of shape (batch_size,3)

            free_cells = torch.zeros((batch_size,3))

            for b in range(batch_size):
                # directions to check
                directions = [
                    self.rotation_matrix_90ccw @ direction[b],
                    direction[b],
                    self.rotation_matrix_90cw @ direction[b]
                ]
                for idd,d in enumerate(directions):

                    k=1
                    cell = head_coord[b] + k*d
                    index = tuple(torch.cat((torch.Tensor([b,0]),cell)).to(int))
                    while (x[index] == 0 or x[index] == -1)\
                    and cell[0] in range(x.shape[2])\
                    and cell[1] in range(x.shape[3]):
                        index = tuple(torch.cat((torch.Tensor([b,0]),cell)).to(int))
                        k+=1
                        cell = head_coord[b] + k*d
                    
                    free_cells[b,idd] = k-1
            
            return torch.cat((free_cells,diff),1) # concatenate along the features dim
            
            
            
 
            
