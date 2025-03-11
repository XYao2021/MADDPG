# -*- coding: utf-8 -*-
""" Created on Wed Oct 12 10:39:48 2022, @Author: xiang_zhang, Univ of Utah
File description
-> module: wireless network
"""
import numpy as np 
from _utils import _antenna_gain_UE_, _antenna_gain_BS_, _compute_distance_,\
                            _beam_align_BS_, _beam_align_UE_, noise_sigma, \
                            dBm2Watt


def sys_setup(n_frames=10000, p_max_dBm=39, bandwidth=500e6, test_channel=0,
                  MSR_BS=20, BW_BS=np.pi/4, MSR_UE=0, BW_UE=2*np.pi, UE_sched=0,
                  pathloss=None, SmallNet=True,
                  fading_dir="fading_Jakes"):
    """INPUT:
        FADING: which trial/realization of channel realization
        fading_dir: "fading_Nakagami" or "fading_Jakes" or "fading_Rayleigh", folder for fading data
    """

    """load BS/UE positions"""
    # if SmallNet:
    #     BS_position = np.load('./data/position/BS_position_small.npy')
    #     UE_position = np.load('./data/position/UE_position_small.npy')
    # else:
    #     BS_position = np.load('./data/position/BS_position.npy')
    #     UE_position = np.load('./data/position/UE_position.npy')
    #
    BS_position = np.array([[25, 75], [75, 75], [75, 25], [25, 25]])
    #
    # UE_position = np.array([[45, 55], [53, 52], [53, 41], [45, 47]])  # UE 1 assigned
    # UE_position = np.array([[22, 56], [74, 57], [80, 48], [30, 15]])  # UE 2 assigned
    UE_position = np.array([[24, 92], [60, 78], [84, 5], [8, 29]])  # UE 3 assigned

    print('Assigned UE position: ', UE_position)

    n_BS = BS_position.shape[0]
    n_UE = UE_position.shape[0]
    n_UE_per_BS = int(n_UE/n_BS)
    # print(n_BS, n_UE)

    """ === load fading data, size=(num_trials,n_slots,n_UE,n_BS)===="""
    # folder = "fading_Nakagami"
    if SmallNet:
        # name = "Fading_small"
        # name = "Fading_small_2"  # with time coefficient
        # name = "Fading_small_1"  # nakagami
        # name = "Fading_jake_new_1.0_10000.0_100"  # nakagami
        # name = "Fading_jake_normal_0.5_10000.0_100"  # nakagami
        # name = "Fading_small_0.0"  # Time correlated Gaussian channel
        name = "Rayleigh_fading_0.5"  # Rayleigh channel
    else:
        name = "Fading"
    Fading_trial = np.load(f"./data/fading/{fading_dir}/{name}.npy")

    """ scheduled UEs"""
    UE_scheduled = [int(UE_sched)]*n_BS

    """ antenna gain and beamwidth """
    antenna_MSR_BS = MSR_BS                        # in dB 
    beamwidth_BS = BW_BS
    antenna_MSR_UE = MSR_UE              # UEs are omnidirectional
    beamwidth_UE = BW_UE
    antenna_gain_min_BS = 1/(beamwidth_BS*10**(antenna_MSR_BS/10)+2*np.pi-beamwidth_BS)
    antenna_gain_max_BS = 10**(antenna_MSR_BS/10)*antenna_gain_min_BS
    antenna_gain_min_UE = 1/(beamwidth_UE*10**(antenna_MSR_UE/10)+2*np.pi - beamwidth_UE)
    antenna_gain_max_UE = 10**(antenna_MSR_UE/10)*antenna_gain_min_UE

    """ === power and noise === """
    p_max = dBm2Watt(p_max_dBm)
    # noise = dBm2Watt(noise_sigma(bandwidth))  #total noise

    N0_dBm = -86.46
    noise = 1e-3 * 10 ** (N0_dBm / 10) / bandwidth

    """ compute distance """
    BS_UE_distance = _compute_distance_(BS_position, UE_position, 0, 0)
    """ === beam alignment ==="""
    Beam_direction_BS = _beam_align_BS_(BS_position=BS_position, UE_position=UE_position,
                                        Scheduled_UE=UE_scheduled, beamwidth=beamwidth_BS)
    Beam_direction_UE = _beam_align_UE_(BS_position=BS_position, UE_position=UE_position,
                                                                 beamwidth=beamwidth_UE)
    """=== compute antenna gain === """
    Antenna_gain_BS = _antenna_gain_BS_(BS_position, UE_position, Beam_direction_BS,
                                                    antenna_MSR_BS, beamwidth_BS)
    Antenna_gain_UE = _antenna_gain_UE_(BS_position, UE_position, Beam_direction_UE,
                                                    antenna_MSR_UE, beamwidth_UE)

    """ === extract direct channels === """
    Fading_mat = Fading_trial[test_channel, :, :, :]   # size=(n_slots,n_UE,n_BS)
    # for i in range(len(Fading_mat[:, 0, 0])):
    #     print(i, Fading_mat[i, :, :])

    Direct_channel = np.zeros([n_frames, n_UE_per_BS, n_BS])
    for slot in range(n_frames):
        for ue in range(n_UE_per_BS):
            for bs in range(n_BS):
                temp = bs*n_UE_per_BS + ue      # UE global index
                Direct_channel[slot, ue, bs] = Fading_mat[slot, temp, bs]

    """generate system info dict """
    sys_dict = dict({})    # sys params
    sys_dict["n_BS"] = n_BS  #
    sys_dict["n_UE"] = n_UE  #
    sys_dict["n_UE_per_BS"] = n_UE_per_BS  #
    sys_dict["BS_UE_distance"] = BS_UE_distance
    sys_dict["BS_position"] = BS_position
    sys_dict["UE_position"] = UE_position
    sys_dict["p_max"] = p_max  #
    sys_dict["noise"] = noise
    sys_dict["Beam_dir_BS"] = Beam_direction_BS
    sys_dict["Beam_dir_UE"] = Beam_direction_UE
    sys_dict["max_ant_gain_BS"] = antenna_gain_max_BS
    sys_dict["min_ant_gain_BS"] = antenna_gain_min_BS
    sys_dict["max_ant_gain_UE"] = antenna_gain_max_UE
    sys_dict["min_ant_gain_UE"] = antenna_gain_min_UE
    sys_dict["ant_gain_BS"] = Antenna_gain_BS 
    sys_dict["ant_gain_UE"] = Antenna_gain_UE 
    sys_dict["Fading_trial"] = Fading_trial
    sys_dict["Fading_mat"] = Fading_mat 
    sys_dict["Direct_channel"] = Direct_channel
    sys_dict["pathloss"] = pathloss 
    sys_dict["bandwidth"] = bandwidth
    sys_dict["UE_sched"] = UE_scheduled  # array,size=(n_BS,)
    sys_dict["MSR_BS"] = antenna_MSR_BS 
    sys_dict["BW_BS"] = beamwidth_BS
    sys_dict["MSR_UE"] = antenna_MSR_UE 
    sys_dict["BW_UE"] = beamwidth_UE
    
    return sys_dict 
