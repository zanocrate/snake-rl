from env import *
from train import *
import torch

env = SnakeEnv(None,15,15)
env.metadata["render_fps"] = 10 # default is 4
action_to_string = {
    0:"UP",
    1:"DOWN",
    2:"RIGHT",
    3:"LEFT"
}

terminated = False

observation, info = env.reset()

policy_net = torch.load('policy_net.pth')

while not terminated:
    
    observation_reshaped = {
        'screen' : observation['screen'].reshape(1,15,15),
        'direction' : observation['direction'].reshape(1,4)
    }

    with torch.no_grad():
        action = int(policy_net(observation_reshaped).argmax(-1).reshape(1,-1)[0][0])

    observation,reward,terminated,info = env.step(action)
    print(observation['screen'])