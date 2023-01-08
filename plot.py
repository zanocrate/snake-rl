# Temporary script to easily plot the latest experiments.

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

files=os.listdir('./data/')
df = {}

for f in files:
    x = np.load('./data/'+f)
    lr,tau=f.split('lr')[1].split('_tau')
    tau = tau.split('.npy')[0]
    lr = float(lr)
    tau = float(tau)
    df[(lr,tau)] = x
# first column is lr, second is tau
df = pd.DataFrame(df)
# rolling mean
df.rolling(500).mean().plot()
plt.show()
