FILENAME = 'CNN2_relative'

LOAD_MODEL = True

# change model class here
from src.models.CNN2_relative import DQN

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

MODEL_PATH = 'trained_models/'+FILENAME+'.pth'

# initialize net to train from zero
dqn = DQN(config['env']['history_length']).to(device)
MODEL_TYPE = dqn.action_space_type

if LOAD_MODEL: dqn.load_state_dict(torch.load(MODEL_PATH))

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
durations,returns = coach.train(config['training']['n_episodes'],config['training']['seed'])

# save resultsx
torch.save(dqn.state_dict(), MODEL_PATH)
np.save('data/'+MODEL_TYPE+'/'+FILENAME+'.durations.npy',durations)
np.save('data/'+MODEL_TYPE+'/'+FILENAME+'.returns.npy',returns)
