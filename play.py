# choose model here
from src.models.CNN2 import DQN
# choose model state dict file here
model_state_path = 'trained_models/CNN2_15000ep.pth'


from src.env import SnakeEnv
from src.coach import epsilon_greedy_policy_qnetwork
import torch
import json
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load shared configs
with open('config.json') as f:
    config = json.load(f)

# set render mode to human
config['env']['render_mode'] = 'human'

# initialize env
env = SnakeEnv(**config['env'])
env.metadata["render_fps"] = 10 # speed up a bit

# load qnetwork
q_net = torch.load(model_state_path)

# run episode
terminated = False
observation, info = env.reset()

while not terminated:
    
    observation = np.expand_dims(observation,0) # add batch size dim for pytorch
    action = epsilon_greedy_policy_qnetwork(observation,0,q_net) # get action
    observation,reward,terminated,info = env.step(action) # step