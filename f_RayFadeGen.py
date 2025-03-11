# -*- coding: utf-8 -*-
""" Created on Thu Feb 16 15:55:48 2023, @Author: xiang_zhang, Univ of Utah
File description
-> Generate Rayleigh fading 
"""
import numpy as np
import matplotlib.pyplot as plt
from _utils import RayleighFading, Time_correlated_Rayleigh


def generate_fading(n_trials, n_slots, n_UE, n_BS, Omega=1, LgNtwk=False,
                    save_output=False):
    input_size = [n_trials, n_slots, n_UE, n_BS]
    Fading = RayleighFading(input_size=input_size, Omega=Omega)
    # print(Fading)
    
    # file_suffix = f"Omega{str(Omega)}"
    # folder_name = "Rayleigh_LargeNet_new" if LgNtwk else "Rayleigh_SmallNet_new"
    # if save_output:
    #     np.save(f"./data/fading/{folder_name}/fading_{file_suffix}_new", Fading)
    # else:
    #     pass
    return Fading


if __name__ == "__main__": 
    # lg = True
    lg = False
    if lg:
        dict = dict({"n_BS": 9, "n_UE": 9, "Omega": 1, "LgNet": not False})
    else:
        dict = dict({"n_BS": 4, "n_UE": 4, "Omega": 1, "LgNet": False})

    # Fading = np.square(generate_fading(n_trials=3,
    #                           n_slots=1000000,
    #                           n_BS=dict["n_BS"],
    #                           n_UE=dict["n_UE"],
    #                           Omega=dict["Omega"],
    #                           LgNtwk=dict["LgNet"],
    #                           save_output=False))
    np.random.seed(42)
    rho = 0.95
    n_slots = 10000000
    n_BS, n_UE = 4, 4
    Fading = np.square(Time_correlated_Rayleigh(n_trials=3,
                                                n_iters=n_slots,
                                                n_UE=n_UE,
                                                n_BS=n_BS,
                                                noise_init=1.0,
                                                rho=rho,
                                                scale=1,
                                                Omega=1))
    
    # """ test """
    np.save("./data/fading/fading_Rayleigh/Rayleigh_fading_{}.npy".format(rho), Fading)
    # print(Fading)
    print(Fading.shape)
    
    # _, ax = plt.subplots()
    # ax.plot(Fading[0,:,0,0][::500])
