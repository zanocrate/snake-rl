from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from src.env import SnakeEnv

import numpy as np
import torch
import tqdm
import os

from copy import deepcopy
from itertools import count

class Coach:

    def __init__(
        self,
        net,
        env_kwargs,
        epsilon,
        tau,
        gamma,
        optimizer_kwargs = {},
        optimizer=None,
        buffer_size = 100000,
        batch_size = 128,
        device='cpu'
    ):

        self.replay_buffer = ReplayMemory(buffer_size)
        self.batch_size = batch_size
        self.env = SnakeEnv(
            **env_kwargs
        )
        self.device = device
        self.policy_net = net
        self.target_net = deepcopy(net)

        self.optimizer = optimizer
        if optimizer is None: 
            self.optimizer = Adam(self.policy_net.parameters(), **optimizer_kwargs)
        
        # RL parameters... where to set these best?
        self.gamma=gamma
        self.epsilon=epsilon # could be a list
        self.tau=tau

        # initialize best net found
        self.best_performance = 1
        self.best_performance_net = deepcopy(self.policy_net)
        


    def optimize_model(self):
        if self.replay_buffer.__len__() < self.batch_size:
            # do not optimize if we do not have enough samples
            return


        data_loader = DataLoader(self.replay_buffer,self.batch_size,shuffle=True)
        batch=next(iter(data_loader))
        


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.logical_not(batch['terminal']).to(self.device)
        non_final_next_states = batch['s2'][non_final_mask].to(self.device)
        state_batch = batch['s'].to(self.device)
        action_batch = batch['a'].to(self.device)
        reward_batch = batch['r'].to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch.numpy()).gather(1,action_batch[:,None])

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # compute
            next_state_values[non_final_mask] = self.target_net(non_final_next_states.numpy()).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def play_episode(self,seed=None):
        """
        Plays and episode, updating the model at every step. If given a seed, will reset every episode to the given seed.
        """
        total_return = 0
        state,_ = self.env.reset(seed)
        for t in count():
            action = epsilon_greedy_policy_qnetwork(state,epsilon=self.epsilon,net=self.policy_net)
            next_state, reward, done, _ = self.env.step(action)

            total_return+=reward

            # Store the transition in memory
            self.replay_buffer._add_sample(state,action,reward,next_state,done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
            if done:
                return (t+1,total_return) # return number of steps of the episode


    def train(self,n_episodes,seed=None):
        episode_durations=np.empty(n_episodes,dtype=int)
        episode_returns=np.empty(n_episodes,dtype=float)
        for i_episode in tqdm.trange(n_episodes):
            
            # if we delete the file loop_flag, the loop exits gracefully
            if 'loop_flag' not in os.listdir(): 
                print('loop_flag not found. Ending loop at episode ',i_episode)
                return episode_durations[:i_episode],episode_returns[:i_episode]
            
            # play an episode, at every step optimize and update replay buffer, and returns the duration
            t,g=self.play_episode(seed=seed)
            episode_durations[i_episode]=t
            episode_returns[i_episode]=g
            if t > self.best_performance:
                self.best_performance = t
                self.best_performance_net = deepcopy(self.policy_net)
        return episode_durations,episode_returns



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

def epsilon_greedy_policy_qnetwork(state,epsilon,net):
    """
    Given state and a Q-network as an input returns the integer of the greedy action.
    """
    n_actions = net.output_layer.out_features 
    if np.random.rand() < epsilon:
        a = np.random.randint(n_actions)
        return a
    else:
        with torch.no_grad(): # disable gradient calculations. useful when doing inference to skip computation
            # find the maximum over the action dim for each batch sample
            # but batch_size is one in inference so expand dims
            q = net(np.expand_dims(state,axis=0)) # q is shaped like (1,n_actions)
            a = q.argmax(1).item() # get the argmax along the actions axis
            return a
      