from env import *
from dqn import *
from trainlib import *

import tqdm
import os
import json

import torch.optim as optim

from itertools import count

with open('config.json') as f:
    config = json.load(f)


N_EPISODES=1000
# "annealing" linear schedule for the exploration parameter
EPSILON_START=0.1
EPSILON_END=0.000001
EPSILONS=np.linspace(EPSILON_START,EPSILON_END,N_EPISODES)

# or is it better to decrease it exponentially?
# also do this to tau?

TAUS=[0.5,0.1,0.0001]
LRS = [100,1,1e-3]


MEMORY_CAPACITY=1000000
BATCH_SIZE = 512

GAMMA=0.99


# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

env = SnakeEnv(**config['env'])

    



# replay memory
REPLAY_MEMORY = ReplayMemory(MEMORY_CAPACITY)

# cross validation
for LR in LRS:
    for TAU in TAUS:
        print('Training with')
        print('LR=',LR)
        print('TAU=',TAU)

        # reinit models
        policy_net = DQN(history_length=config['env']['history_length']).to(device) # theta i
        target_net = DQN(history_length=config['env']['history_length']).to(device) # theta i-1, used to compute the target yi
        # initially clone them
        target_net.load_state_dict(policy_net.state_dict())

        # Adam optimizer
        optimizer = optim.Adam(policy_net.parameters(), lr=LR)

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
        np.save('./data/durations_lr'+str(LR)+'_tau'+str(TAU),np.array(episode_durations))