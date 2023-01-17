N_EPISODES=5000
SEED=None # set to None to randomize starts
FILENAME = 'NN_custom_features'

# change model class here
from src.models.NN_custom_features import DQN

from src.coach import * # also imports Coach class

import torch
import json
import os

# create loop flag
os.close(os.open("loop_flag", os.O_CREAT))

# load configs
with open('config.json') as f:
    config = json.load(f)

# choose device here
device = 'cpu'

# initialize net to train from zero
dqn = DQN(config['env']['history_length']).to(device)
MODEL_TYPE = dqn.action_space_type

# initialize coach instance with training and policy parameters
coach = Coach(
    net=dqn,
    env_kwargs=config['env'],
    epsilon=config['training']['epsilon'],
    tau=config['training']['tau'],
    gamma=config['training']['gamma'],
    optimizer_kwargs=config['training']['optimizer'],
    buffer_size=config['training']['buffer_size'],
    batch_size=config['training']['batch_size'],
    device=device
)

# run the loop
durations,returns = coach.train(N_EPISODES,SEED)

# save resultsx
torch.save(dqn.state_dict(), 'trained_models/'+FILENAME+'.pth')
np.save('data/'+MODEL_TYPE+'/'+FILENAME+'.durations.npy',durations)
np.save('data/'+MODEL_TYPE+'/'+FILENAME+'.returns.npy',returns)
