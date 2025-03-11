import numpy as np
import matplotlib.pyplot as plt
from _utils import move_avg

# names = ["Data_MADDPG_F20_stop1.0_noise0.8",
#          "Data_MADDPG_F20_stop0.5_noise0.8",
#          "Data_MADDPG_F20_stop0.2_noise0.8",
#          "Data_MADDPG_NF_noise0.8",
#          "Data_MADDPG_F20_stop1.0_noise0.2",
#          "Data_MADDPG_F20_stop0.5_noise0.2",
#          "Data_MADDPG_F20_stop0.1_noise0.2",
#          "Data_MADDPG_NF_noise0.2"
#          ]
# dir = "./data/saved_data/"

# names = ["Data_MADDPG_F20_S0.5_N0.5",
#          "Data_MADDPG_F20_S0.5_N0.1",
#          "Data_MADDPG_F20_S0.5_N0",
#          "Data_MADDPG_NF_S0.5_N0.5",
#          "Data_MADDPG_NF_S0.5_N0.1",
#          "Data_MADDPG_NF_S0.5_N0" ]
# dir = "./data/saved_data/folder/"  #data directory

# names = ["Data_MADDPG_NF_S0.5_N0.1_1",
#         "Data_MADDPG_NF_S0.5_N0.1_2",
#         "Data_MADDPG_NF_S0.5_N0.1_3",
#         "Data_MADDPG_NF_S0.5_N0.1_4"]
#         # "Data_MADDPG_NF_S0.5_N0.1_5"]
#
# dir = "./data/saved_data/folder1/"  #data directory


names = ["Data_MADDPG_S0.5_F50",
        "Data_MADDPG_S1_F20",
        "Data_MADDPG_S1_NF"]

dir = "./data/saved_data/folder2/"  #data directory



data = dict({})
for i in range(len(names)):
    data[names[i]] = np.load(f"{dir+names[i]}.npy")

lens = [] #data time length
for i in range(len(names)):
    lens.append(data[names[i]].shape[1])

min_len=min(lens)

_, ax = plt.subplots()
for i in range(len(names)):
    ax.plot(range(min_len),move_avg(data[names[i]][0,0:min_len],window_size=1),label=names[i],lw=0.5)  #WMMMSE

ax.set_xlabel("Slots")
ax.set_ylabel("Avg throughput per BS (bps/sec/Hz)")
ax.set_ylim([6, None])
ax.legend(labels=names)
ax.grid()
plt.savefig("./figure/fig_performance.pdf")
plt.show()
