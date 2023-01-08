from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.optim as optim

class ReplayMemory(Dataset):

    def __init__(self,max_capacity):
        
        # init max_capacity
        self.max_capacity = max_capacity
        # init samples as empty lists
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
            random_drop = np.random.randint(0,self.max_capacity)
            self.s.pop(random_drop)
            self.a.pop(random_drop)
            self.r.pop(random_drop)
            self.s2.pop(random_drop)


def random_policy():
    return np.random.randint(4)


def epsilon_greedy_policy(state,epsilon,net):
    if isinstance(state,dict):
        raise NotImplementedError()
    else:
        if np.random.rand() < epsilon:
            return random_policy()
        else:
            with torch.no_grad(): # disable gradient calculations. useful when doing inference to skip computation
                # find the maximum over the action dim for each batch sample
                # but batch_size is one in inference so expand dims
                q = net(np.expand_dims(state,axis=0))
                a = q.argmax(1).item() 
                return a # is an int
            

