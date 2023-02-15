from src.env import SnakeEnv
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.registry import register_env
import json

with open('config.json') as f:
    config = json.load(f)

# https://docs.ray.io/en/latest/rllib/rllib-env.html
def env_creator(env_config):
    return SnakeEnv(**env_config)  # return an env instance

register_env("snake", env_creator)

algo = (DQNConfig()
    .environment(env="snake",env_config=config['env'],render_env=False) # give the env https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-environments
    .framework('torch') # here we specify the framework for the model https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-framework-options
    .rollouts() # rollout means episode in ray lingo https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-rollout-workers
    .training(gamma=0.7,lr=0.0001,train_batch_size=512) #https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-training-options
    .build()
)

algo.train()