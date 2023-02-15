import numpy as np
from torch.utils.data import Dataset
import torch

class ReplayMemory(Dataset):

    action_space_types = ['absolute','relative']

    """
    Replay Memory Dataset.
    Add a transition (s,a,r,s') with .__add_sample()
    """
    def __init__(self,max_capacity,state_shape,action_space_type,augment=True):
        
        self.augment = augment
        assert action_space_type in self.action_space_types
        self.action_space_type = action_space_type

        # initialize memory as empty arrays
        self.max_capacity = max_capacity

        self.buffer = {
        's': torch.empty((max_capacity,*state_shape)),
        'a': torch.empty(max_capacity,dtype=int),
        'r': torch.empty(max_capacity),
        's2': torch.empty((max_capacity,*state_shape)),
        'terminal': torch.empty(max_capacity)
        }
        # bookkeeping current index to fill
        self.index_to_fill = 0
    
    def __len__(self):
        return self.index_to_fill # the index to fill tells us how many samples are there
    
    def __getitem__(self,idx):
        
        sample = {
            key : value[idx] for (key,value) in self.buffer.items()
        }

        return sample

    def _add_sample(self,s,a,r,s2,terminal):
        """
        s and s2 are numpy arrays of shape (history_len,width,height)
        a is an int
        r is float
        """
        
        if self.augment:
            samples = self._agument_sample((s,a,r,s2,terminal))
        else:
            samples = [(s,a,r,s2,terminal)]

        for sample in samples:

            if self.index_to_fill >= self.max_capacity: 
                # the next index to fill would exceed capacity
                # then replace a random sample of the replay memory
                index = np.random.randint(self.max_capacity)
            else:
                # fill the current index with random sample
                index = self.index_to_fill
                self.index_to_fill += 1

            self.buffer['s'][index] = torch.from_numpy(sample[0].copy())
            self.buffer['a'][index] = sample[1]
            self.buffer['r'][index] = sample[2]
            self.buffer['s2'][index] = torch.from_numpy(sample[3].copy())
            self.buffer['terminal'][index] = sample[4]

    def _agument_sample(self,sample):
        """
        Takes (s,a,r,s2,terminal) sample and returns a list of [(s,a,r,s2,terminal)] samples, on for each of game symmetries.
        """

        samples = [sample]

        if self.action_space_type == 'relative':
            # only rotate the s,s2 numpy arrays. they are shaped (k,w,h) so rotate on the plane of 
            for k in range(1,4):
                s=np.rot90(sample[0],k=k,axes=(1,2))
                a=sample[1]
                r=sample[2]
                s2=np.rot90(sample[3],k=k,axes=(1,2))
                terminal=sample[4]
                samples.append((s,a,r,s2,terminal))
        elif self.action_space_type == 'absolute':
            
            for k in range(1,4):
                s=np.rot90(sample[0],k=k,axes=(1,2))
                # also rotate the action; in the env, they are ordered as successive ccw rotation, so just add 1 to the index
                a = (sample[1] + 1) % 4
                r=sample[2]
                s2=np.rot90(sample[3],k=k,axes=(1,2))
                terminal=sample[4]
                samples.append((s,a,r,s2,terminal))
        
        return samples

    def save(self,path):
        """
        Save the replay memory samples to a given directory path.
        """

        torch.save(self.buffer,path)
    
    def load(self,path):
        """
        Load samples from a directory path.
        """
        
        self.buffer = torch.load(path)