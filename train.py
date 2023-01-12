N_EPISODES=20000
SEED=567 # set to None to randomize starts
save_name = 'CNN2_relative'
# change model here
from src.models.CNN2_relative import DQN

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


durations,returns = coach.train(N_EPISODES,SEED)

torch.save(dqn.state_dict(), 'trained_models/'+str(save_name)+'.pth')
np.save('data/'+str(save_name)+'.durations.npy',durations)
np.save('data/'+str(save_name)+'.returns.npy',returns)
