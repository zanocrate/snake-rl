from env import *

env = SnakeEnv('human',10,10)
env.metadata["render_fps"] = 10 # default is 4
action_to_string = {
    0:"UP",
    1:"DOWN",
    2:"RIGHT",
    3:"LEFT"
}

terminated = False

observation, info = env.reset()

while not terminated:
    action = env.action_space.sample()
    observation,reward,terminated,info = env.step(action)