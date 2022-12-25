from env import *
from train import *
import torch

env = SnakeEnv('human',15,15)
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
        actions = policy_net(observation_reshaped)
        print(actions)
        action = int(actions.argmax(-1))
	
    print(action)
    observation,reward,terminated,info = env.step(action)
    
