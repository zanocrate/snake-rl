# change model here
from src.models.CNN1 import DQN

N_EPISODES=100

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


coach.train(N_EPISODES)

torch.save(dqn.state_dict(), 'trained_models/CNN1_100ep.pth')