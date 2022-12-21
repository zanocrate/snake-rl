from env import * # also import numpy as np and gym and pygame
import torch
from torch import nn

WIDTH = 10
HEIGHT = 10

env = SnakeEnv(
    render_mode=None,
    width=WIDTH,
    height=HEIGHT,
    periodic=True,
    food_reward=1,
    terminated_penalty=-1
)

class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.screen_features_ = nn.Sequential(
                  nn.Conv2d(1,1),
                  nn.ReLU(),
                  nn.Linear(64,64*2),
                  nn.ReLU(),
                  nn.Linear(64*2,action_space_dim)
                )

    def forward(self, x):
        x = x.to(device)
        return self.linear(x)
