from src.env import SnakeEnv
import numpy as np
import json

with open('config.json') as f:
	config = json.load(f)

env = SnakeEnv(**config['env'])

while True:
	seed = np.random.randint(1000)
	obs,_=env.reset(seed)
	food_coord = np.argwhere(obs[0]==-1)[0]
	snake_coord = np.argwhere(obs[0]==1)[0]
	d=abs(food_coord-snake_coord)
	if (d<=2).all():
		print('Seed found:')
		print(seed)
		print(obs[0])
		break
