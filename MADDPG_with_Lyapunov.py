"""MADDPG-based power control, all BS share
[June 01]
Periodically feed WMMSE powers into replay buffer to boost DRL learning
"""
import time
import os
import json
import numpy as np
import cvxpy as cp
import torch
import torch as T
from matplotlib import pyplot as plt

from _maddpg import MultiAgentReplayBuffer, MADDPG
from _PHY_sys import sys_setup
from _utils import _pathloss_, _compute_interference_, _move_avg_, _scale_data_,\
                   obs_list_to_state_vector, unit_map, gen_init_power,\
                    _composite_channel_, dBm2Watt, Utility_function
from main_WMMSE_OneShot import WMMSE
from f_PlotRewardDist import plot_rwd_hist


def main_MADDPG_w_Lyapunov(n_frames=1000, n_slots=1000, n_trials=3, action_noise="Gaussian",
         noise_init=1.0, noise_min=0.05, noise_decay=1-5e-5,
         test_channel=0, alpha=1.0, beta=0.0,
         MSR=20, beamwidth=np.pi/4, power_max_dBm=39, bandwidth=500e6,
         UE_sched=0,
         hidden_size=None, lr_AC=None,
         ShareRwd=True, ScaleInterf=True, LoadModel=False,
         buffer_size=50_0000, batch_size=64, discount_factor=0.9,
         AC_dims=None,
         TrainMode=True, pathloss=None,
         InitPower="zero", Init_power=None,
         LearnFreq=1, FeedFreq=10, GD_per_slot=1,
         SmallNet=True, WMMSE_stop=0.5, target_update_freq=1, window_size=200,
         V=3000, T_f=0.5, T_b=0.5, T_unit=1e-3, W=1, P_avg=1, g_max=1e12):
    r"""
    INPUT:
        action_dim: action dim\n
        action_noise: exploration noise, "Gaussian" or "Uniform"\n
        noise_init: initial noise std/range\n
        noise_min: min std for Gaussian, range for uniform\n
        alpha, beta: reward weights\n
        MSR, beamwidth: BS antenna MSR, beamwidth\n
        test_channel: int, channel realization index (under same distribution)\n
        UE_sched: scheduled UE, int\n
        hidden_size: NN w two hidden layers\n
        lr_AC: actor, critic lr\n
        ShareRwd: True if team rwd is used\n
        ScaleInterf: True if interf is scaled\n
        load_chkpt: True is the initial model params are loaded from pretrained ones\n
        LoadModel: True if pretrained models need to be loaded\n
        buffer_size: replay buffer size\n
        batch_size: mini-batch size\n
        AC_dims: [actor_dim, critic dim], number of dims of agent local obs, global state\n
        train_mode: True if in train mode, else put in test mode (no learning)\n
        InitPower: "zero","full" or "predefined", initial power mode\n
        Init_power: size=(n_BS,1), specify if InitPower ="predefined"\n
        FeedFreq: int, use WMMSE powers every this number of slots\n
        LearnFreq:int, train actor/critic every this number of slots\n
        GD_per_slot: int, number of GD for actor/critic per slot\n
        WMMSE_stop: [0,1], stop WMMSE feeding after n_slots*WMMSE_stop slots
    OUTPUT:
        Data: (slide window averaged) avg throughput, array, size=(3,n_slots)
    """
    start_time = time.time()

    """System setup: generate channel related parameters."""
    sys_params_dict = sys_setup(n_frames=n_frames*n_slots,
                                p_max_dBm=power_max_dBm,
                                bandwidth=bandwidth,
                                test_channel=test_channel,
                                MSR_BS=MSR,
                                BW_BS=beamwidth,
                                MSR_UE=0,
                                BW_UE=2*np.pi,
                                UE_sched=UE_sched,
                                pathloss=pathloss,
                                SmallNet=SmallNet,
                                fading_dir="fading_Rayleigh")  # fading_dir = fading_Jakes / fading_Nakagami / fading_Rayleigh

    """ //// Learning params ///// """
    gamma = discount_factor
    """ ///// Scheduling params //////"""
    UE_sched = sys_params_dict["UE_sched"]  # sched UEs
    """ ///// PHY params ///// """
    pathloss = sys_params_dict["pathloss"]
    p_max = sys_params_dict["p_max"]
    noise = sys_params_dict["noise"]
    n_BS = sys_params_dict["n_BS"]
    # print(n_BS)
    n_UE_per_BS = sys_params_dict["n_UE_per_BS"]
    Fading_mat = sys_params_dict["Fading_mat"]
    print('Fading_mat: ', Fading_mat.shape)
    #small-scale fading at test_channel,size=(n_slots,n_UE,n_BS)
    Direct_channel = sys_params_dict["Direct_channel"]
    antenna_gain_max_BS = sys_params_dict["max_ant_gain_BS"]  # antenna gain & beamwidth
    antenna_gain_max_UE = sys_params_dict["max_ant_gain_UE"]
    """ BS/UE distance """
    BS_UE_distance = sys_params_dict["BS_UE_distance"]
    """ antenna gain """
    Antenna_gain_BS = sys_params_dict["ant_gain_BS"]
    Antenna_gain_UE = sys_params_dict["ant_gain_UE"]
    """/////// MADDPG params ///////// """
    actor_dim, critic_dim = AC_dims
    actor_dims = [actor_dim for _ in range(n_BS)]
    critic_dims = critic_dim
    n_actions = 1  # action dim
    tau = 0.005  # target nets soft update coefficient
    # tau = 0.01
    """ max interf at sched UEs for scaling interf"""
    p = p_max * np.ones(n_BS)
    Fading_max = np.max(Fading_mat, axis=0)  # size=(n_UE, n_BS),avg fading coeff over time
    Interf_max, _ = _compute_interference_(UE_sched, p, BS_UE_distance, Antenna_gain_BS,
                                           Antenna_gain_UE, Fading_max, pathloss)
    Interf_max += noise

    """/////////////// TRIAL LOOP ////////////////////"""
    """ interf, rwd & power over trials"""
    Interf_profile_trial = np.zeros([n_trials, n_BS, n_frames])
    Reward_profile_trial = np.zeros([n_trials, n_BS, n_frames])
    U_trail = np.zeros([n_trials, n_BS, n_frames])
    Power_profile_trial = np.zeros([n_trials, n_BS, n_frames])
    Throughput_profile_trial = np.zeros([n_trials, n_BS, n_frames])

    Power_avg_profile_trail = np.zeros([n_trials, n_BS, n_frames])
    H_ji_profile_trail = np.zeros([n_trials, n_BS, n_frames])

    """begin trial loop"""
    for trial_idx in range(n_trials):
        # """Lyapunov queue initialization"""
        # Z_i = np.zeros([n_UE_per_BS, n_BS])
        # H_ji = np.zeros([n_UE_per_BS, n_BS])
        # X_ji = np.zeros([n_UE_per_BS, n_BS])  # achieved throughput of UE j, assume one UE per BS

        """ create agents """
        T.manual_seed(42)       # keep all weight init fixed
        MADDPG_agents = MADDPG(actor_dims=actor_dims,
                               critic_dims=critic_dims,
                               n_agents=n_BS,
                               n_actions=n_actions,
                               fc1=hidden_size[0],
                               fc2=hidden_size[1],
                               fc3=hidden_size[2],
                               alpha=lr_AC[0],
                               beta=lr_AC[1],
                               gamma=gamma,
                               tau=tau,
                               action_noise=action_noise,
                               noise_init=noise_init,
                               noise_min=noise_min,
                               noise_decay=noise_decay,
                               chkpt_dir="tmp/maddpg")  # return a list containing all agents
        """ load checkpoint"""
        if LoadModel and len(os.listdir("./tmp/maddpg")) > 1:  # exclude the hidden file .DS_store
            MADDPG_agents.load_checkpoint()
        memory = MultiAgentReplayBuffer(max_size=buffer_size,
                                        critic_dims=critic_dims,
                                        actor_dims=actor_dims,
                                        n_actions=n_actions,
                                        n_agents=n_BS,
                                        batch_size=batch_size)
        """/////////// SLOT LOOP /////////////"""
        """ interf, reward & power over slots """
        Interf_profile = np.zeros([n_BS, n_frames], dtype=np.float32)
        Interf_profile_ = np.zeros([n_BS, n_frames], dtype=np.float32)  # decision point interf
        Indiv_interf_profile = np.zeros([n_BS, n_BS, n_frames], dtype=np.float32)
        # i-th row: component interf measured at BS i from each other BS (g(t)p(t))
        Indiv_interf_profile_ = np.zeros([n_BS, n_BS, n_frames], dtype=np.float32)  # DP interf g(t+1)p(t)
        Reward_profile = np.zeros([n_BS, n_frames], dtype=np.float32)
        U_i = np.zeros([n_BS, n_frames], dtype=np.float32)

        Power_profile = np.zeros([n_BS, n_frames], dtype=np.float32)  # store powers in all slots
        score_history = []  # avg rwd history
        factor = 0.95  # tanh clip to [-factor, factor], must be the same as that in _maddpg.py
        """ init obs/state & actions """
        obs_list_ = [np.zeros(actor_dim) for _ in range(n_BS)]  # list of arrays
        state_ = np.zeros(critic_dim)

        "Queue Initialization"
        # max_throughput = T_f * W * np.log2(1 + g_max * p_max)
        throughput_initial_value = 0
        power_initial_value = 0
        Z_i = power_initial_value * np.zeros(n_BS)
        H_ji = throughput_initial_value * np.zeros(n_BS)
        X_ji = np.zeros(n_BS)

        """//// begin slot loop ///"""
        # slot_idx = 0
        # a = 0
        for frame_idx in range(n_frames):
            power_profile_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            reward_profile_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            Interf_profile_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            Interf_profile_frame_ = np.zeros([n_BS, n_slots], dtype=np.float32)
            Indiv_interf_profile_frame = np.zeros([n_BS, n_BS, n_slots], dtype=np.float32)
            Indiv_interf_profile_frame_ = np.zeros([n_BS, n_BS, n_slots], dtype=np.float32)
            U_i_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            X_ji_frame = np.zeros(n_BS, dtype=np.float32)

            for slot_idx in range(n_slots):
                if slot_idx == 0:  # first slot uses initial power
                    if frame_idx == 0:
                        pass
                    else:
                        Init_power = Power_profile[:, frame_idx-1]
                    obs_list = [np.zeros(actor_dim) for _ in range(n_BS)]
                    state = np.zeros(critic_dim)
                    # initial action/power
                    actions = gen_init_power(n_BS=n_BS,
                                             mode=InitPower,
                                             predef_power=Init_power,
                                             tanh_factor=factor)  # [-factor,factor]
                else:
                    obs_list = obs_list_  # next state of last slot is current state of curr slot
                    state = state_
                    """ ///choose action/// """
                    if slot_idx % FeedFreq == 0 and slot_idx < WMMSE_stop*n_slots:  # Why set FeedFreq = n_slots? -> Not apply WMMSE action
                        CompCh = _composite_channel_(BS_UE_distance,
                                                        Antenna_gain_BS,
                                                        Antenna_gain_UE,
                                                        Fading_mat[frame_idx, :, :],
                                                        pathloss)
                        power_wmmse, _ = WMMSE(Composite_channel=CompCh,
                                               sched_UEs=UE_sched,
                                               power_max=p_max,
                                               noise=noise,
                                               n_iters=2000,
                                               weights=np.ones(n_BS))  #WMMSE powers,size=(n_BS,)
                        actions = gen_init_power(n_BS=n_BS,
                                                 mode="predefined",
                                                 predef_power=power_wmmse/p_max,
                                                 tanh_factor=factor)  #convert [0,1] to [-factor,factor]
                    else:  # maddpg power
                        actions = MADDPG_agents.choose_action(np.array(obs_list, dtype=np.float32))  # current power choice

                # print(actions)
                Tanh_actor = True  # use tanh activation at actor output
                # Tanh_actor = False
                if Tanh_actor:  # tanh actor
                    power_profile_frame[:, slot_idx] = np.abs(np.array([p_max * (actions[i].item() + factor)
                                                                        / (2 * factor) for i in range(n_BS)]))
                else:  # sigmoid actor
                    power_profile_frame[:, slot_idx] = np.array([p_max * actions[i].item() for i in range(n_BS)])
                # print(slot_idx, 'observations: ', obs_list)
                # print(frame_idx, slot_idx, 'Power selection: ', power_profile_frame[:, slot_idx])

                """compute interf at current slot (channel+power) """
                interf_vec, Interf_mat = _compute_interference_(UE_sched,
                                                                power_profile_frame[:, slot_idx],
                                                                BS_UE_distance,
                                                                Antenna_gain_BS,
                                                                Antenna_gain_UE,
                                                                Fading_mat[frame_idx*n_slots + slot_idx-1, :, :],
                                                                pathloss)

                # interf_vec, Interf_mat = _compute_interference_(UE_sched,
                #                                                 power_profile_frame[:, slot_idx],
                #                                                 BS_UE_distance,
                #                                                 Antenna_gain_BS,
                #                                                 Antenna_gain_UE,
                #                                                 Fading_mat[frame_idx, :, :],
                #                                                 pathloss)
                # print(frame_idx, 'direct interference: ', interf_vec)
                # print(frame_idx, 'mapping interference: ', Interf_mat, '\n')
                # """idea verification END"""

                Interf_profile_frame[:, slot_idx] = interf_vec
                Indiv_interf_profile_frame[:, :, slot_idx] = Interf_mat  # component interf g_ki(t)p_ki(t)

                """ compute decision point interf (at the beginning of next slot)"""
                interf_vec_, Interf_mat_ = _compute_interference_(UE_sched,
                                                                 power_profile_frame[:, slot_idx],
                                                                 BS_UE_distance,
                                                                 Antenna_gain_BS,
                                                                 Antenna_gain_UE,
                                                                 Fading_mat[min(frame_idx*n_slots+slot_idx, frame_idx*n_slots+n_slots-1), :, :],
                                                                 pathloss)
                #
                # interf_vec_, Interf_mat_ = _compute_interference_(UE_sched,
                #                                                   power_profile_frame[:, slot_idx],
                #                                                   BS_UE_distance,
                #                                                   Antenna_gain_BS,
                #                                                   Antenna_gain_UE,
                #                                                   Fading_mat[min(frame_idx+1, n_frames-1), :, :],
                #                                                   pathloss)

                Interf_profile_frame_[:, slot_idx] = interf_vec_
                Indiv_interf_profile_frame_[:, :, slot_idx] = Interf_mat_  # component interf g_ki(t+1)p_ki(t)

                # print(frame_idx, slot_idx, 'Hji: ', H_ji)
                # print(frame_idx, slot_idx, 'Zi: ', Z_i)
                """reward function need to be replaced by lyapunov optimization framework (alpha/beta)"""
                for bs in range(n_BS):
                    """ compute reward (throughput/sec/Hz) """
                    ue_sch = UE_sched[bs]
                    ue_g = bs * n_UE_per_BS + ue_sch

                    SINR = power_profile_frame[bs, slot_idx] * antenna_gain_max_BS * antenna_gain_max_UE * \
                           _pathloss_(BS_UE_distance[ue_g][bs], pathloss) * \
                           Direct_channel[frame_idx*n_slots + slot_idx-1, ue_sch, bs] / (Interf_profile_frame[bs, slot_idx] + noise * W)

                    # SINR = power_profile_frame[bs, slot_idx] * antenna_gain_max_BS * antenna_gain_max_UE * \
                    #        _pathloss_(BS_UE_distance[ue_g][bs], pathloss) * \
                    #        Direct_channel[frame_idx, ue_sch, bs] / (
                    #                    Interf_profile_frame[bs, slot_idx] + noise * W)

                    """Update alpha and beta according to current Lyapunov queue"""
                    alpha = H_ji[bs] * T_b / max(max(H_ji), 0.0000000001)  # T_b is (fixed) block time, need to be defined previously
                    beta = Z_i[bs] * T_b / max(max(Z_i), 0.0000000001)
                    # print(slot_idx, bs, alpha, beta)
                    X_ji_frame[bs] += T_unit * W * np.log2(1 + SINR)  # T_b and W need to be defined previously
                    reward_profile_frame[bs, slot_idx] = alpha * W * np.log2(1+SINR) - beta * power_profile_frame[bs, slot_idx]
                    # if reward_profile_frame[bs, slot_idx] < 0:
                    #     a += 1

                    # if slot_idx == 0:
                    #     U_i_frame[bs, slot_idx] = 0
                    # else:
                    #     U_i_frame[bs, slot_idx] = np.log2(X_ji[bs])
                # print(reward_profile_frame[:, slot_idx])
                # ratios = torch.clip(torch.tensor(reward_profile_frame[:, slot_idx]), -1, 1)
                normalized_reward = torch.tensor(reward_profile_frame[:, slot_idx] / max(torch.tensor(reward_profile_frame[:, slot_idx]).max(), 0.0000000001))
                # ratios = torch.tensor(normalized_reward)
                # print(slot_idx, reward_profile_frame[:, slot_idx])
                # ratios = torch.softmax(torch.tensor(reward_profile_frame[:, slot_idx]), dim=0)
                ratios = torch.softmax(normalized_reward, dim=0)
                # print(frame_idx, slot_idx, 'ratios: ', ratios)

                # target = T.tensor(reward_profile_frame[:, slot_idx])
                # if slot_idx > 0:
                #     mean = target.mean()
                #     std = target.std()
                #     target = (target - mean) / std
                #     # print(slot_idx, target, mean, std, Reward_profile[:, slot_idx])
                # ratios = T.softmax(target, dim=0)
                # print(frame_idx, slot_idx, 'rewards: ', reward_profile_frame[:, slot_idx])
                # print(frame_idx, slot_idx, 'normalized rewards: ', normalized_reward)

                """ next state (i.e., current state for slot t+1) """
                obs_list_ = []  # construct local obs
                for i in range(n_BS):
                    local_obs_ = []
                    local_obs_.append(power_profile_frame[i, slot_idx])  # self power group

                    local_obs_.append(ratios[i])

                    local_obs_.append(sum(reward_profile_frame[:, slot_idx]))  # locally shared rwd grp
                    local_obs_ += list(reward_profile_frame[:, slot_idx])  # throughput of all BSs
                    # local_obs_.append(sum(normalized_reward))  # locally shared rwd grp
                    # local_obs_ += list(normalized_reward)  # throughput of all BSs
                    """direct channel grp """
                    local_obs_.append(Direct_channel[frame_idx * n_slots + slot_idx - 1, UE_sched[i], i])  # g(t)
                    local_obs_.append(Direct_channel[min(frame_idx * n_slots + slot_idx + 1, frame_idx * n_slots + n_slots - 1), UE_sched[i], i])  # g(t+1)

                    # local_obs_.append(Direct_channel[frame_idx, UE_sched[i], i])  # g(t)
                    # local_obs_.append(Direct_channel[min(frame_idx + 1, n_frames - 1), UE_sched[i], i])  # g(t+1)
                    """total interf grp """
                    temp = Interf_profile_frame[i, slot_idx] + noise if not ScaleInterf else \
                        _scale_data_(Interf_profile_frame[i, slot_idx] + noise, Interf_max[i], 0.0)
                    temp_ = Interf_profile_frame_[i, slot_idx] + noise if not ScaleInterf else \
                        _scale_data_(Interf_profile_frame_[i, slot_idx] + noise, Interf_max[i], 0.0)
                    local_obs_.append(temp)
                    local_obs_.append(temp_)
                    """ component interf from neighboring BSs """
                    if ScaleInterf:
                        temp = [_scale_data_(Indiv_interf_profile_frame[i, j, slot_idx] + noise,
                                             Interf_max[i], 0.0) for j in range(n_BS) if j != i]  # g(t)p(t)
                        temp_ = [_scale_data_(Indiv_interf_profile_frame_[i, j, slot_idx] + noise,
                                              Interf_max[i], 0.0) for j in range(n_BS) if j != i]  # g(t+1)p(t)
                    else:
                        temp = [Indiv_interf_profile_frame[i, j, slot_idx] + noise for j in range(n_BS) if j != i]
                        temp_ = [Indiv_interf_profile_frame_[i, j, slot_idx] + noise for j in range(n_BS) if j != i]
                    local_obs_ += temp
                    local_obs_ += temp_

                    obs_list_.append(np.array(local_obs_))
                    # print(i, local_obs_)
                state_ = obs_list_to_state_vector(obs_list_)  # flatten list (of 1D arrays) to 1D array
                """store experience """
                done = False
                rwd_list = [sum(reward_profile_frame[:, slot_idx]) for _ in range(n_BS)] \
                    if ShareRwd else [reward_profile_frame[i, slot_idx] for i in range(n_BS)]
                # rwd_list = [sum(normalized_reward) for _ in range(n_BS)] \
                #     if ShareRwd else [normalized_reward[i] for i in range(n_BS)]
                memory.store_transition(obs_list, state, actions, rwd_list, obs_list_, state_, done)

                # print(frame_idx, slot_idx, 'Reward', reward_profile_frame[:, slot_idx])
                # print(frame_idx, slot_idx, 'states', state_)
                """ learn w a minibatch   """
                if TrainMode is True:  # Org: Train actor / critic / target_actor / target_critic together --> should be different
                    if slot_idx % LearnFreq == 0:
                        for _ in range(GD_per_slot):
                            MADDPG_agents.learn(memory, slot_idx=slot_idx, target_freq=target_update_freq)
                else:
                    pass
                # plt.plot(X_ji_frame)
                # plt.savefig('throughput for {} frame'.format(frame_idx))
            # print(frame_idx, X_ji_frame)

            Power_profile[:, frame_idx] = power_profile_frame[:, -1]
            Interf_profile[:, frame_idx] = Interf_profile[:, -1]
            Reward_profile[:, frame_idx] = reward_profile_frame[:, -1]
            U_i[:, frame_idx] = Utility_function(X_ji_frame, utility_power=0.6)
            X_ji = X_ji_frame
            # score_history.append(np.mean(Reward_profile[:, frame_idx]))
            score_history.append(np.mean(U_i[:, frame_idx]))

            """Solve Lyapunov optimization (Auxiliary Variables)"""
            x = cp.Variable(n_BS)
            objective = cp.Minimize(-V * cp.sum(cp.log(x) / cp.log(2).value) + cp.sum(H_ji.T @ x))
            constraints = [x <= T_f * W * (cp.log(1 + g_max * p_max)/cp.log(2)).value, 0 <= x]
            problem = cp.Problem(objective, constraints)
            # problem.solve(verbose=True, solver=cp.MOSEK)  # verbose is using for debug
            problem.solve()
            x_optimal = x.value
            # print(slot_idx, 'x_optimal: ', x_optimal)

            """Queue Update"""
            for bs in range(n_BS):
                """Update Lyapunov queues, P_avg and T_f need to be defined previous"""
                Z_i[bs] = max(Z_i[bs] + T_f * Power_profile[bs, frame_idx] - T_f * P_avg, 0)
                # Z_i[bs] = max(Z_i[bs] + T_unit * sum(power_profile_frame[bs, :]) - T_f * P_avg, 0)
                H_ji[bs] = max(H_ji[bs] + x_optimal[bs] - X_ji[bs], 0)  # Need to debug, has problem: solver's problem?

            # print(frame_idx, 'X_ji: ', X_ji)
            # print(frame_idx, 'x_optimal: ', x_optimal)
            "Lyapunov Debug information"
            print(frame_idx, 'power', Power_profile[:, frame_idx])
            print(frame_idx, 'Z_i(t)', Z_i)
            print(frame_idx, 'H_ji(t)', H_ji, '\n')

            """Store the time related parameters"""
            for bs in range(n_BS):
                if frame_idx == 0:
                    Power_avg_profile_trail[trial_idx, bs, frame_idx] = Power_profile[bs, frame_idx]
                else:
                    Power_avg_profile_trail[trial_idx, bs, frame_idx] = sum(Power_profile[bs, :frame_idx]) / len(Power_profile[bs, :frame_idx])
            H_ji_profile_trail[trial_idx, :, frame_idx] = H_ji

            # print('\n')
            """print progress """
            if frame_idx % 2 == 0:
                print("--------------------------------------")
                print("Trial:%d|frame:%d/%d\n>Rwd:%.2f|Trail_avg_rwd:%.2f" % (
                    trial_idx, frame_idx, n_frames, score_history[frame_idx], np.mean(score_history[:frame_idx])))
                print("--------------------------------------")

        """ save checkpoint """
        MADDPG_agents.save_checkpoint()
        """ record reward & power over trials"""
        Interf_profile_trial[trial_idx, :, :] = Interf_profile  #size=(num_rep,n_BS,n_slots)
        Reward_profile_trial[trial_idx, :, :] = Reward_profile
        U_trail[trial_idx, :, :] = U_i
        # Throughput_profile_trial[trial_idx, :, :] = Throughput_profile
        Power_profile_trial[trial_idx, :, :] = Power_profile

    """ print exe time """
    dur = time.time() - start_time
    print("Time:%.2f secs total,%.2f secs/slot" % (dur, dur/n_frames))
    # print(U_trail)
    """ /// moving avg w fixed window ////// """
    # Rwd_avg_BS = Reward_profile_trial.mean(axis=1)  # avg over BSs,size=(num_rep,n_slots)
    Rwd_avg_BS = U_trail.mean(axis=1)  # avg over BSs,size=(num_rep,n_slots)

    Rwd_over_trials = np.zeros([n_trials, n_frames])
    for i in range(n_trials):
        # Rwd_over_trials[i, :] = _move_avg_(Rwd_avg_BS[i, :], window_size=window_size)
        for j in range(n_frames):
            Rwd_over_trials[i, j] = sum(Rwd_avg_BS[i, :j + 1]) / len(Rwd_avg_BS[i, :j + 1])

    """save plot data, rows 0,1,2-> mean, max, min """
    Data = np.zeros([4, n_frames])
    Data[0, :] = Rwd_over_trials.mean(axis=0)
    Data[1, :] = Rwd_over_trials.max(axis=0)
    Data[2, :] = Rwd_over_trials.min(axis=0)
    Data[3, :] = Rwd_over_trials.std(axis=0)

    return Data, Reward_profile_trial[-1], Power_profile_trial[-1], Power_avg_profile_trail[-1], H_ji_profile_trail[-1]  # last trial rewards, powers,size=(n_BS,n_slots)


def main(IsSmallNet=True, start_noise=1, noise_decay=1-1e-5, wmmse_stop=1, FeedFreq=100, GDFreq=1, target_update_freq=1, window_size=200, n_frames=1000, n_slots=500, bandwidth=500e6):
     r"""
     :param IsSmallNet: True if 4 BSs is used
     :param n_slots:
     :param start_noise: Gaussian expl noise initial variance
     :param noise_decay: decay rate of exploration noise
     :param wmmse_stop: in [0,1], when to stop feeding wmmse powers
     :param FeedFreq: feed wmmse power every this number of slots
     :param GDFreq: do this number of GD (Gradient Descent) per slot
     :return:
     """
     p_max_dBm = 40
     TrainMode = True
     LoadModel = not True  # train mode off if model is loaded
     min_noise = 0

     """Lyapunov related parameters"""
     V = 5000  # Example
     # T_unit = 1e-3
     T_b = 0.5  # 1 block per frame
     T_f = 0.5
     T_unit = T_b / n_slots
     # T_unit = 1e-3
     # T_b = n_slots * T_unit
     # T_f = n_slots * T_unit
     # W = 20e3
     # W = 400e6
     W = 1
     P_avg_dBm = 38.13  # 38.13
     # P_avg_dBm = 35
     P_avg = 1e-3 * 10 ** (P_avg_dBm / 10)
     g_max = 1e12
     # g_max = 1e20
     # pathloss = ["dual_slope", 2, 2, 1]
     pathloss = ["dual_slope", 4, 4, 1]
     # pathloss = ["uniform", 4, 2, 1]
     # beamwidth = np.pi / 4
     beamwidth = np.pi / 6

     "Check current states performance without Lyapunov framework"

     "UE 1"
     # Learning_rate_pair = [1e-8, 5e-8]  # Not exactly
     #
     # Network_layers = [256, 128, 64]
     # Network_layers = [240, 120, 64]
     # Network_layers = [200, 100, 64]

     "UE 2"
     # Learning_rate_pair = [5e-7, 5e-6]  # worked
     # Network_layers = [240, 120, 64]

     # TODO 1
     # Learning_rate_pair = [5e-7, 3e-6]
     # Network_layers = [240, 120, 64]
     # TODO 2
     # Network_layers = [228, 114, 64]
     # Learning_rate_pair = [1e-6, 5e-6]

     # Learning_rate_pair = [1e-6, 5e-6]  # worked  It seems model will be diverged around 370 frames
     # Network_layers = [240, 120, 64]  # worked

     # "UE 3"
     # Learning_rate_pair = [1e-5, 1e-4]  # worked well
     # Network_layers = [200, 100, 64]

     "UE 3, test"
     Learning_rate_pair = [1e-4, 1e-3]
     # Learning_rate_pair = [1e-5, 1e-4]
     Network_layers = [256, 128, 64]
     # Network_layers = [200, 100, 64]

     print(Learning_rate_pair)
     print(Network_layers)

     # AC_dims = [17 - 1, 4 * 17 - 4]
     AC_dims = [17, 4 * 17]

     """read simulation configurations"""
     with open('parameters.txt', 'r') as file:
         json_str = file.read()
     params = json.loads(json_str)  # Convert the JSON string to a dictionary
     Data_MADDPG, Rwd_MADDPG, Power_MADDPG, Power_avg, H_ji = main_MADDPG_w_Lyapunov(
                            n_trials=1,
                            action_noise="Gaussian",
                            noise_init=start_noise,
                            noise_min=min_noise,
                            noise_decay=noise_decay,
                            test_channel=0,
                            # change this to change channel realizations (same distribution)
                            MSR=params["MSR"],
                            beamwidth=params["BW"],
                            UE_sched=params["UE_sched"],
                            power_max_dBm=p_max_dBm,
                            bandwidth=bandwidth,
                            # hidden_size=[512, 256, 256],  # for 9 BSs
                            hidden_size=Network_layers,  # for 4 BSs
                            # hidden_size=[256, 128, 64],  # for 4 BSs (previous choice)
                            # hidden_size=[128, 64, 64],
                            lr_AC=Learning_rate_pair,
                            FeedFreq=FeedFreq,
                            LearnFreq=1,
                            GD_per_slot=GDFreq,
                            ShareRwd=True,
                            ScaleInterf=True,  # scale interference
                            LoadModel=LoadModel,
                            buffer_size=100_0000,  # 100_0000, max size for replay buffer
                            batch_size=128,  # 64
                            AC_dims=AC_dims,
                            TrainMode=TrainMode,
                            pathloss=pathloss,
                            InitPower="predefined",
                            SmallNet=IsSmallNet,
                            WMMSE_stop=wmmse_stop,
                            Init_power=np.zeros(4),
                            target_update_freq=target_update_freq,
                            window_size=window_size,
                            n_frames=n_frames,
                            n_slots=n_slots,
                            V=V, T_f=T_f, T_b=T_b, T_unit=T_unit, W=W, P_avg=P_avg, g_max=g_max)

     time_finished = time.strftime("%H:%M:%S", time.localtime())
     np.save('npy_data/MADDPG_data_{}_{}_{}.npy'.format(time_finished, Learning_rate_pair, Network_layers), Data_MADDPG)

     MSR, BW, n_BS = params["MSR"], params["BW"], len(Power_MADDPG[:, 0])
     label = "MADDPG throughput.\n" + f"FeedFreq:{FeedFreq}\nGDFreq:{GDFreq}\n" \
             + f"MSR,BW:{MSR}dB,pi/{np.pi / BW:.1f}\n" \
             + f"Expl_noise_init:{start_noise}\n" \
             + f"Expl_noise_decay:{noise_decay}"

     plt.plot(Data_MADDPG[0], label=label, lw=0.8)
     plt.ylabel('Average throughput (bits/sec/Hz per BS)')
     plt.xlabel('Frames')
     # plt.ylim([min(Data_MADDPG[2]) - 0.5, max(Data_MADDPG[1]) + 0.5])
     plt.ylim([0, max(Data_MADDPG[1]) + 0.1])
     plt.legend(loc='lower right')
     plt.grid()
     # plt.savefig('figure/MADDPG_fig/MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     fig, ax = plt.subplots(n_BS, 1, sharex=True)  # plot powers
     fig.suptitle(f"MADDPG power over time")
     for bs in range(n_BS):
         ax[bs].plot(Power_MADDPG[bs, -int(n_frames / 1):-1], label=f"BS {bs}", lw=0.8)
         ax[bs].set_title(f'BS {bs}', fontsize=8)
         ax[bs].set_ylim([0, dBm2Watt(p_max_dBm)])
         ax[bs].grid()
         if bs == n_BS - 1:
             ax[bs].set_xlabel('Slots')
     # plt.savefig('figure/Power_fig/Power_instant_MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     fig, ax = plt.subplots(n_BS, 1, sharex=True)  # plot powers
     fig.suptitle(f"MADDPG H_ji over time")
     for bs in range(n_BS):
         ax[bs].plot(H_ji[bs, -int(n_frames / 1):-1], label=f"BS {bs}", lw=0.8)
         ax[bs].set_title(f'BS {bs}', fontsize=8)
         ax[bs].grid()
         if bs == n_BS - 1:
             ax[bs].set_xlabel('Frames')
     # plt.savefig('figure/Lyapunov_fig/H_ji_MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     fig, ax = plt.subplots(n_BS, 1, sharex=True)
     fig.suptitle(f"MADDPG P_avg over time")
     for bs in range(n_BS):
         ax[bs].plot(Power_avg[bs, -int(n_frames / 1):-1], label=f"BS {bs}", lw=0.8)
         ax[bs].set_title(f'BS {bs}', fontsize=8)
         ax[bs].plot([P_avg for i in range(n_frames)])
         ax[bs].set_ylim([0, dBm2Watt(p_max_dBm)])
         ax[bs].grid()
         if bs == n_BS - 1:
             ax[bs].set_xlabel('Frames')
     # plt.savefig('figure/Lyapunov_fig/Power_avg_MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     """plot reward histogram"""
     # plot_rwd_hist(Rwd_MADDPG)


if __name__ == "__main__":
    n_frames = 1000
    n_slots = 500
    main(IsSmallNet=True,
         n_slots=n_slots,
         n_frames=n_frames,
         start_noise=1,
         bandwidth=400e6,
         noise_decay=1-2e-9,
         wmmse_stop=1,
         FeedFreq=n_slots,
         GDFreq=1,
         target_update_freq=5,
         window_size=250)
# target 2048
