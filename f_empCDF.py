# -*- coding: utf-8 -*-
""" Created on Mon Feb 27 11:34:32 2023, @Author: xiang_zhang, Univ of Utah
File description
-> draw empirical CDF of data 
"""
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# load data 
names = ["FP", "WMMSE", "Random", "Full Reuse", "MADDPG"]
# names = ["FP"]
data = dict({})
LgNtwk = not True 
if not LgNtwk:
    for name in names:
        data[name] = np.load(f"./data/plot_data/Data_{name}.npy")   #4 BS net
else:
    for name in names:
        data[name] = np.load(f"./saved_data_figure/5BS_net/Data_{name}.npy") #5 BS net
    
    
# generate cdf
ECDF = dict({})
for name in names:
    ECDF[name] = sm.distributions.ECDF(data[name][0])
    
# Create plot
ls = ["--", "-.", ":", ":", "-"]
fig, ax = plt.subplots()
for i, name in enumerate(names):
    if name == "MADDPG":
        ax.plot(ECDF[name].x, ECDF[name].y, linestyle=ls[i], label ="Proposed")
    else:
        ax.plot(ECDF[name].x, ECDF[name].y, linestyle=ls[i], label =name)

# Set labels and title
ax.legend()
ax.set_xlim([2,None])
ax.set_xlabel('Throughput (bps/Hz per BS)')
ax.set_ylabel('Empirical CDF')
# ax.set_title('Empirical Cumulative Distribution Function')

# Show plot
plt.savefig("./figure/fig_ecdf.pdf")
plt.show()

