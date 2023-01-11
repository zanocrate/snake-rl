N_EPISODES=5000

# change model here
from src.models.CNN2 import DQN

from src.coach import * # also imports Coach class

import torch
import json

with open('config.json') as f:
    config = json.load(f)

device = 'cpu'

dqn = DQN(config['env']['history_length']).to(device)

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


durations = coach.train(N_EPISODES)

torch.save(dqn.state_dict(), 'trained_models/CNN2_1024bs.pth')
np.save('data/CNN2_1024_bs.npy',durations)
