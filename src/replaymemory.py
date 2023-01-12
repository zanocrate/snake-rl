import numpy as np
from torch.utils.data import Dataset

class ReplayMemory(Dataset):
    """
    Replay Memory Dataset.
    Add a transition (s,a,r,s') with .__add_sample()
    """
    def __init__(self,max_capacity):
        
        self.max_capacity = max_capacity
        self.s = []
        self.a = []
        self.r = []
        self.s2 = []
        self.terminal = []
    
    def __len__(self):
        return len(self.s) # return length of one of the lists
    
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
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s2.append(s2)
        self.terminal.append(terminal)
        
        if self.__len__() > self.max_capacity:
            random_drop = np.random.randint(0,self.max_capacity) # drop a random entry 
            self.s.pop(random_drop)
            self.a.pop(random_drop)
            self.r.pop(random_drop)
            self.s2.pop(random_drop)