from env import *
import numpy as np

import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from itertools import count

# game parameters

WIDTH = 15
HEIGHT = 15
HISTORY_LENGTH = 4

# loop parameters
LOAD_CHECKPOINT = False
CHECKPOINT_EVERY=2
CHECKPOINT_PATH=os.getcwd()+'/checkpoint/'

N_EPISODES=10
# "annealing" linear schedule for the exploration parameter
EPSILON_START=0.1
EPSILON_END=0.000001
EPSILONS=np.linspace(EPSILON_START,EPSILON_END,N_EPISODES)

# or is it better to decrease it exponentially?
# also do this to tau?

TAU=0.1

MEMORY_CAPACITY=1000000

LR = 1e-3
BATCH_SIZE = 512

GAMMA=0.99


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




env = SnakeEnv(
    render_mode=None,
    width=WIDTH,
    height=HEIGHT,
    periodic=False,
    observation_type=1, # this returns only the screen
    history_length=HISTORY_LENGTH
)

class DQN(nn.Module):
    """
    Class for the Neural Network of the Q value function.
    """
    def __init__(self,history_length):
        super(DQN, self).__init__()

        # a first layer with narrow filters to identify walls, corners, dead ends...
        # returns (w-2)x(h-2) when padding is 'valid'
        self.conv1 = nn.Conv2d(
            in_channels=history_length, # the number of states fed
            out_channels=15, # the number of kernels
            kernel_size=3, # a single int means square kernel
            stride=1, # it's the default: no skipping
            padding='valid', # no padding
            # dilation=1, #this expands the filter without augmenting the parameters... to identify spatially large patterns with less parameters
            # groups=1, # determines the connection between input channels and output channels
            # bias=True
            # padding_mode='circular' # revert to default which is zeroes... circular breaks it
        )
        
        # non linear activation
        self.activation1 = nn.ReLU()
        
        # a larger kernel size to identify bigger configurations of smaller patterns?
        # since we are at a 13x13 screen, maybe do a 5x5 kernel
        self.conv2 = nn.Conv2d(
            in_channels=15,
            out_channels=15,
            kernel_size=7,
            stride=4,
            padding='valid'
        )

        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(
                in_channels=15,
                out_channels=15,
                kernel_size=2,
                padding='valid',
                bias=True
            )
        
        self.activation3 = nn.ReLU()


        # now the feed forward layers
        self.ffl1 = nn.Linear(
            15,
            256
        )

        self.ffl_activation1 = nn.ReLU()

        # final layer
        self.ffl2= nn.Linear(
            256,
            4
        )


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):

        x = torch.Tensor(x)
        x = self.activation1(self.conv1(x))
        x = self.activation2(self.conv2(x))
        x = self.activation3(self.conv3(x))
        x = self.ffl_activation1(self.ffl1(x.flatten(start_dim=1))) # do not drop the batch dimension
        x = self.ffl2(x)

        return x


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
            

def play_episode(policy,env,memory=None,**policy_kwargs):
    """
    Play an episode following a given policy, and if ReplayMemory is given adds (s,a,r,s2) transitions to it.
    Args:
    -----
        policy : func, receives a state and returns an action
        memory : ReplayMemory instance
    Returns:
    -----
        episode : list of [(s,a,r,s2,done)] tuples
    """
    done = False
    episode = []
    while not done:
        s,_=env.reset()
        
        a = policy(s,**policy_kwargs)

        s2,r,done,_=env.step(a)

        if memory is not None:
            memory._add_sample(s,a,r,s2,done)
        episode.append((s,a,r,s2))
    return episode


if LOAD_CHECKPOINT:
    policy_net = torch.load(CHECKPOINT_PATH+'policy_net.pth').to(device)
    target_net = torch.load(CHECKPOINT_PATH+'target_net.pth').to(device)
else:
    # thetai and thetai-1
    policy_net = DQN(history_length=HISTORY_LENGTH).to(device) # theta i
    target_net = DQN(history_length=HISTORY_LENGTH).to(device) # theta i-1, used to compute the target yi
    # initially clone them
    target_net.load_state_dict(policy_net.state_dict())

# Adam optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=LR)
# replay memory
REPLAY_MEMORY = ReplayMemory(MEMORY_CAPACITY)

def optimize_model():
    if REPLAY_MEMORY.__len__() < BATCH_SIZE:
        # do not optimize if we do not have enough samples
        return
    # data loader
    data_loader = DataLoader(REPLAY_MEMORY,BATCH_SIZE,True)

    batch=next(iter(data_loader))


    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.logical_not(batch['terminal']).to(device)
    non_final_next_states = batch['s2'][non_final_mask].to(device)
    state_batch = batch['s'].to(device)
    action_batch = batch['a'].to(device)
    reward_batch = batch['r'].to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.numpy()).gather(1,action_batch[:,None])

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # compute
        next_state_values[non_final_mask] = target_net(non_final_next_states.numpy()).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

episode_durations=[]

for i_episode in tqdm.trange(N_EPISODES):

    state,_ = env.reset()
    for t in count():
        action = epsilon_greedy_policy(state,epsilon=EPSILONS[i_episode],net=policy_net)
        next_state, reward, done, _ = env.step(action)



        # Store the transition in memory
        REPLAY_MEMORY._add_sample(state,action,reward,next_state,done)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        if done:
            episode_durations.append(t+1)
            break

    if (i_episode+1) % CHECKPOINT_EVERY == 0: 
        index = (i_episode+1) // CHECKPOINT_EVERY
        np.save(CHECKPOINT_PATH+'training_episode_durations.'+str(index)+'.npy',np.array(episode_durations))
        torch.save(policy_net,CHECKPOINT_PATH+'policy_net.'+str(index)+'.pth')
        torch.save(target_net,CHECKPOINT_PATH+'target_net.'+str(index)+'.pth')
