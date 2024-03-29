from src.env import SnakeEnv
from src.replaymemory import *

from copy import deepcopy
from itertools import count
import os
import json

from tqdm import trange

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


def train_loop(json_f,model=None):

    env_kwargs = json_f['env']
    config = json_f['training']
    logging_config = json_f['logging']
        
    ################### LOGGING

    run_path = os.path.join(logging_config['log_dir'],logging_config['run_name'])
    writer = SummaryWriter(run_path)

    # save run parameters
    os.makedirs(run_path, exist_ok=True)
    with open(os.path.join(run_path,'params.json'), 'w') as outfile:
        json.dump(config,outfile,indent=6)
    with open(os.path.join(run_path,'env.json'), 'w') as outfile:
        json.dump(env_kwargs,outfile,indent=6)

    ################### CONFIGS


    augment_dataset = config['augment_dataset']
    n_episodes = config['n_episodes']
    buffer_size = config['buffer_size']
    seed = config['seed']

    epsilon_start,epsilon_end = config['epsilon']['start_end'] # is a list of two epsilons
    if config['epsilon']['space'] == 'linear':
        epsilons = np.linspace(epsilon_start,epsilon_end,n_episodes)
    elif config['epsilon']['space'] == 'log':
        epsilons = np.logspace(epsilon_start,epsilon_end,n_episodes, base = 10)

    # other parameters

    # threshold for gradient clipping
    max_norm = config['clipping_threshold']
    

    ################### TRAINING

    # env init
    env = SnakeEnv(**env_kwargs)

    state_shape = (env_kwargs['history_length'],env_kwargs['width'],env_kwargs['height'])
    replay_buffer = ReplayMemory(buffer_size,state_shape,env_kwargs['action_space_type'],augment_dataset)
    if logging_config['load_buffer_path'] is not None: replay_buffer.load(logging_config['load_buffer_path'])

    if model is None:
        # if model is not provided, load from src/model.py
        from src.model import DQN
        # initialize Net
        model = DQN(state_shape,env_kwargs['action_space_type'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = deepcopy(model)
    target_net = deepcopy(model)

    policy_net.to(device)
    target_net.to(device)

    # optimizer
    optimizer = Adam(policy_net.parameters(), lr = config['lr'])

    ############################################################ EPISODES LOOP

    returns = np.empty(n_episodes,dtype=float)

    for i_episode in trange(n_episodes):

        
        # play the episode
        returns[i_episode] = 0
        best_return = -999999999999
        state,_ = env.reset(seed)

        # initialize state list; first history_length random frames
        states = [state[k+1] for k in range(-env_kwargs['history_length'],0)]

        for t in count():

            #################################################### EPSILON GREEDY POLICY

            n_actions = policy_net.output_layer.out_features 
            if np.random.rand() < epsilons[i_episode]:
                action = np.random.randint(n_actions)
            else:
                with torch.no_grad(): # disable gradient calculations. useful when doing inference to skip computation
                    # find the maximum over the action dim for each batch sample
                    # but batch_size is one in inference so expand dims
                    q = policy_net(torch.tensor(np.expand_dims(state,axis=0),dtype=torch.float,device=device)) # q is shaped like (1,n_actions)
                    action = q.argmax(1).item() # get the argmax along the actions axis

        
            next_state, reward, done, _ = env.step(action)

            returns[i_episode]+=reward
            
            # Store the transition in memory
            replay_buffer._add_sample(state,action,reward,next_state,done)

            # Move to the next state
            state = next_state
            states += [state[0]]

            #################################################### TRAINING
            
            # only if we have enough samples
            if replay_buffer.__len__() >= config['batch_size']:

                data_loader = DataLoader(replay_buffer,config['batch_size'],shuffle=True)
                
                # batch is a dict of tensors for s,a,r,s2,terminal
                batch=next(iter(data_loader))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.logical_not(batch['terminal'].to(device)).to(device)
                non_final_next_states = batch['s2'].to(device)[non_final_mask]
                state_batch = batch['s'].to(device)
                action_batch = batch['a'].to(device)
                reward_batch = batch['r'].to(device)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = policy_net(state_batch).gather(1,action_batch[:,None])

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(config['batch_size'], device=device)
                with torch.no_grad():
                    # compute
                    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * config['gamma']) + reward_batch

                # Compute loss
                criterion = torch.nn.MSELoss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), max_norm)
                optimizer.step()


            #################################################### UPDATE Q_i -> Q_(i-1)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*config['tau'] + target_net_state_dict[key]*(1-config['tau'])
            target_net.load_state_dict(target_net_state_dict)
            
            #################################################### SAVE METRICS

            if done:
                
                writer.add_scalar("T",t, i_episode)
                writer.add_scalar("G",returns[i_episode], i_episode)
                writer.add_scalar("epsilon",epsilons[i_episode],i_episode)
                # writer.add_graph(policy_net,input_to_model=state)

                # VIDEO RENDERING
                # vid_tensor: (N,T,C,H,W)
                # adding frames dimension
                vid_tensor = np.stack(states) # (T,H,W)
                # adding rgb channels dimension
                vid_tensor = np.stack((vid_tensor,vid_tensor,vid_tensor),axis=1)
                # adding batch dimension
                vid_tensor = np.expand_dims(vid_tensor,axis=0) # (1,T,1,W,H)
                
                #set RGB values
                vid_tensor[:,:,0,:,:] = False # R channel is screen
                vid_tensor[:,:,1,:,:] = (vid_tensor[:,:,1,:,:] == -1) # G channel is food
                vid_tensor[:,:,2,:,:] = (vid_tensor[:,:,2,:,:] > 0) # B channel is snake
                vid_tensor = vid_tensor.astype(int)
                writer.add_video("episode",vid_tensor,i_episode,fps=10)
                
                break
        
        # save state of policy net after each episode

        torch.save(policy_net.state_dict(), os.path.join(run_path,'policy_net.pth'))


#################################################### SAVE HYPERPARAMETERS


    writer.close()

    




if __name__ =='__main__':

    with open('config.json') as f:
        json_f = json.load(f)

    train_loop(json_f)
