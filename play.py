# choose model here
from src.models.NN_custom_features import DQN
# choose model state dict file here
MODEL_STATE_FILENAME = "NN_custom_features.pth"
# set seed to None for random
SEED = 567

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

model_state_path = 'trained_models/'+config['env']['action_space_type']+'/'+MODEL_STATE_FILENAME

# initialize env
env = SnakeEnv(**config['env'])
env.metadata["render_fps"] = 10 # speed up a bit

# load qnetwork
q_net = DQN(config['env']['history_length'])
q_net_state_dict = torch.load(model_state_path)
q_net.load_state_dict(q_net_state_dict)

# run episode
terminated = False
observation, info = env.reset(SEED)

while not terminated:
    
    # observation = np.expand_dims(observation,0) # add batch size dim for pytorch
    action = epsilon_greedy_policy_qnetwork(observation,0.1,q_net) # get action
    observation,reward,terminated,info = env.step(action) # step
    print(reward)
