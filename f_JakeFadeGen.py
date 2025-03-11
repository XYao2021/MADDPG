"""
Generate small-scale fading according to Jake's correlated channel model
"""
import numpy as np
from _utils import JakesFading_Gaussian as JFG
from matplotlib import pyplot as plt

def generate_fading(n_trials=1,n_slots=100,n_BS=10,n_UE=10,rho=0.9,scale=1.0,SaveOutput=False,
                    folder="fading_Jakes",file="Fading",rand_seed=42):
    r"""Generate temporal correlated Gaussian fading h^2
        INPUT:
            rho: correlation coefficient in Jakes model
            scale: controls Gaussian variance in each slot
            SaveOutput: bool, save output if True
        OUTPUT:
            Fading_sqrd: size=(n_trials, n_slots,n_UE,n_BS)
            """
    Fading_sqrd = np.square(JFG(n_trials=n_trials,
                                n_iters=n_slots,
                                n_UE=n_UE,
                                n_BS=n_BS,
                                noise_init=1.0,
                                rho=rho,
                                scale=scale,
                                rand_seed=rand_seed))
    # if SaveOutput:
        # np.save(f"./data/fading/{folder}/{file}", Fading_sqrd)
    # else:
    #     pass
    return Fading_sqrd


if __name__ == "__main__":
    SmallNet = True
    rho = 0.1
    if SmallNet:
        file = "Fading_small"
        n_BS, n_UE = 4, 4
    else:
        file = "Fading"
        n_BS, n_UE = 9, 9
    if True:
        Fading = generate_fading(n_trials=3,
                                 n_slots=1000000,
                                 n_BS=n_BS,
                                 n_UE=n_UE,
                                 rho=rho,
                                 SaveOutput=True,
                                 file=file, rand_seed=42)
    print(Fading)

    np.save(f"./data/fading/fading_Jakes/Fading_small_{rho}.npy", Fading)
    print(f"Data shape:", Fading.shape)

    # _, ax = plt.subplots()
    # ax.plot(Fading[0,:1000,0,0],'.-',label=f"rho={rho}")
    # ax.legend()
    # plt.show()
