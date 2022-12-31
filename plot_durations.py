import matplotlib.pyplot as plt
import numpy as np
import os
import glob

path=os.getcwd()+'/checkpoint/*.npy'
files = glob.glob(path)
n_files = len(files)

print('Loading training_episode_durations.'+str(1)+'.npy')
x=np.load('./checkpoint/training_episode_durations.'+str(1)+'.npy')

for i in range(1,n_files):
    print('training_episode_durations.'+str(i+1)+'.npy')
    x=np.concatenate((x,np.load('./checkpoint/training_episode_durations.'+str(i+1)+'.npy')),axis=None)


