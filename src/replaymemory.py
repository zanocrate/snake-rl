import numpy as np
from torch.utils.data import Dataset
import torch

class ReplayMemory(Dataset):
    """
    Replay Memory Dataset.
    Add a transition (s,a,r,s') with .__add_sample()
    """
    def __init__(self,max_capacity,state_shape):
        
        # initialize memory as empty arrays
        self.max_capacity = max_capacity
        self.s = torch.empty((max_capacity,*state_shape))
        self.a = torch.empty(max_capacity,dtype=int)
        self.r = torch.empty(max_capacity)
        self.s2 = torch.empty((max_capacity,*state_shape))
        self.terminal = torch.empty(max_capacity)

        # bookkeeping current index to fill
        self.index_to_fill = 0
    
    def __len__(self):
        return self.index_to_fill # the index to fill tells us how many samples are there
    
    def __getitem__(self,idx):
        
        sample = {
            's' : self.s[idx],
            'a' : self.a[idx],
            'r' : self.r[idx],
            's2' : self.s2[idx],
            'terminal' : self.terminal[idx]
        }

        return sample

    def _add_sample(self,s,a,r,s2,terminal):
        """
        s and s2 are numpy arrays of shape (history_len,width,height)
        a is an int
        r is float
        """
        
        if self.index_to_fill >= self.max_capacity: 
            # the next index to fill would exceed capacity
            # then replace a random sample of the replay memory
            index = np.random.randint(self.max_capacity)
        else:
            # fill the current index with random sample
            index = self.index_to_fill
            self.index_to_fill += 1
        self.s[index] = torch.from_numpy(s)
        self.a[index] = torch.from_numpy(np.array(a))
        self.r[index] = torch.from_numpy(np.array(r))
        self.s2[index] = torch.from_numpy(s2)
        self.terminal[index] = torch.from_numpy(np.array(terminal))

    def save(self,path):
        """
        Save the replay memory samples to a given directory path.
        """
        if path[-1] != '/': path+='/'

        torch.save(self.s,path+'s.pt')
        torch.save(self.a,path+'a.pt')
        torch.save(self.r,path+'r.pt')
        torch.save(self.s2,path+'s2.pt')
        torch.save(self.terminal,path+'terminal.pt')
    
    def load(self,path):
        """
        Load samples from a directory path.
        """
        if path[-1] != '/': path+='/'

        self.s=torch.load(path+'s.pt')
        self.a=torch.load(path+'a.pt')
        self.r=torch.load(path+'r.pt')
        self.s2=torch.load(path+'s2.pt')
        self.terminal=torch.load(path+'terminal.pt')