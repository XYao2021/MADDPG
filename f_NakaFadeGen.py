"""
File description
-> Generate time-correlated Nakagami distribution,
    store to folder "./data/fading/"
"""
import numpy as np
from _utils import temporal_Nakagami, JakesFading_Nakagami
from scipy.stats import nakagami, uniform

def generate_fading(n_trials, n_slots, n_BS, n_UE, m=50, Omega=1.0, rho=0.5, save_output=False):
    """Generate and store time-correlated Nakagami fading h^2
        INPUT:
            m: Nakagami shape param, type:int (larger m,smaller variance)
            Omega: =1 fixed
            rho: correlation coefficient
        OUTPUT:
            Fading: fading array (h^2), size=(n_trials, n_slots,n_UE,n_BS)
            save_output: save output to fading folder
    """
    Fading = np.square(temporal_Nakagami(n_trials=n_trials,
                                         n_slots=n_slots,
                                         n_BS=n_BS,
                                         n_UE=n_UE,
                                         m=m,
                                         Omega=Omega,
                                         rho=rho))  # normal distribution
    # Fading = np.square(JakesFading_Nakagami(n_trials=n_trials,
    #                                      n_iters=n_slots,
    #                                      n_BS=n_BS,
    #                                      n_UE=n_UE,
    #                                      nu=m,
    #                                      rho=rho))
    folder = "fading_Nakagami"
    if save_output:
        # np.save(f"./data/fading/{folder}/Fading_jake_small_{rho}_{m}_{Omega}", Fading)  #save to folder
        np.save(f"./data/fading/{folder}/Fading_jake_normal_{rho}_{m}_{Omega}", Fading)  # save to folder
    else:
        pass

    return Fading


"""generate fading"""
if __name__ == "__main__":
    n_BS, n_UE, rho = 4, 4, 0.5
    n_trails = 3
    n_slots = 100000
    m = 1e4
    omega = 100
    if True:
        Fading = generate_fading(n_trials=n_trails,
                                 n_slots=n_slots,
                                 n_BS=n_BS,
                                 n_UE=n_UE,
                                 m=1e4,  # 50: channel nearly no change, smaller m means channel varied more serve. m=1 equals to rayleigh fading
                                 rho=rho,
                                 save_output=True,
                                 Omega=100)
        # Fading = nakagami.rvs(nu=m, scale=np.sqrt(omega/m), size=[n_trails, n_slots, n_BS, n_UE])**2
        # print(Fading)
        # folder = "fading_Nakagami"
        # np.save(f"./data/fading/{folder}/Fading_jake_new_{rho}_{m}_{omega}", Fading)
        # np.save(f"./data/fading/{folder}/Fading_small_1.npy", Fading)
    # Fading = np.load (f"./data/fading/fading_Nakagami/Fading_small.npy")
    print(f"Data shape:", Fading.shape)
