from env import *
from dqn import *
from trainlib import *

import tqdm
import os
import json



from itertools import count

with open('config.json') as f:
    config = json.load(f)


N_EPISODES=5000
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
