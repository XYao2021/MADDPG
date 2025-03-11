"""plot the reward distribution (histogram) of users"""
import numpy as np
import matplotlib.pyplot as plt
import time


def plot_rwd_hist(Data):
    r"""
    :param: Data: input data, size(n_BS,n_slots)
    :return: None
    """
    Data = np.array(Data)
    n_BS = Data.shape[0]
    fig, ax = plt.subplots(n_BS,sharex=True)
    for i in range(n_BS):
        ax[i].hist(Data[i], bins=50, edgecolor='black',label=f"User {i}")  # Adjust 'bins' as needed
        ax[i].set_ylabel("Frequency")

    ax[n_BS-1].set_xlabel("Reward (bits/Hz/sec)")
    fig.suptitle("Reward distribution")
    time_finished = time.strftime("%H:%M:%S", time.localtime())
    plt.savefig(f"./figure/Reward_fig/fig_RwdHist_{time_finished}.pdf")
    plt.show()


if __name__ == "__main__":
    Data = np.random.rand(3, 1000)
    plot_rwd_hist(Data)
