"""MADDPG-based power control, all BS share
[June 01]
Periodically feed WMMSE powers into replay buffer to boost DRL learning
"""
import time
import os
import json
import numpy as np
import cvxpy as cp
import torch as T
from matplotlib import pyplot as plt

from _maddpg import MultiAgentReplayBuffer, MADDPG
from _PHY_sys import sys_setup
from _utils import _pathloss_, _compute_interference_, _move_avg_, _scale_data_,\
                   obs_list_to_state_vector, unit_map, gen_init_power,\
                    _composite_channel_, dBm2Watt, find_state_percentile, Utility_function
from main_WMMSE_OneShot import WMMSE
from f_PlotRewardDist import plot_rwd_hist


def main_QL_w_Lyapunov(n_frames=1000, n_slots=1000, n_trials=3, test_channel=0, alpha=1.0, beta=0.0,
                          MSR=20, beamwidth=np.pi/4, power_max_dBm=39, bandwidth=500e6, UE_sched=0,
                          discount_factor=0.9, pathloss=None, SmallNet=True, window_size=200,
                          V=3000, T_f=0.5, T_b=0.5, T_unit=1e-3, W=1, P_avg=1, g_max=1e12,
                          P_q=40, I_q=16, learning_rate=0.1, number_train_epoch=2000, epsilon=0.05):
    r"""
    OUTPUT:
        Data: (slide window averaged) avg throughput, array, size=(4,n_slots)
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
    # noise = 1e-3 * 10 ** (-84.46 / 10) / W
    n_BS = sys_params_dict["n_BS"]
    # print(n_BS)
    n_UE_per_BS = sys_params_dict["n_UE_per_BS"]
    Fading_mat = sys_params_dict["Fading_mat"]
    print('Fading_mat: ', Fading_mat.shape)
    #small-scale fading at test_channel,size=(n_slots,n_UE,n_BS)
    Direct_channel = sys_params_dict["Direct_channel"]
    print(Direct_channel.shape)
    antenna_gain_max_BS = sys_params_dict["max_ant_gain_BS"]  # antenna gain & beamwidth
    antenna_gain_max_UE = sys_params_dict["max_ant_gain_UE"]
    """ BS/UE distance """
    BS_UE_distance = sys_params_dict["BS_UE_distance"]
    """ antenna gain """
    Antenna_gain_BS = sys_params_dict["ant_gain_BS"]
    Antenna_gain_UE = sys_params_dict["ant_gain_UE"]
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

    """ create Q table """
    discount_factor = discount_factor
    learning_rate = learning_rate
    epsilon = epsilon
    P_MAX = dBm2Watt(power_max_dBm)
    P_step = P_MAX / (P_q - 1)
    P_quan = [i * P_step for i in range(P_q)]
    I_step = 100 / I_q
    I_quan = [(i + 1) * I_step for i in range(I_q)]

    "Q table pre-train session"
    channel_map_length = len(Fading_mat)
    # number_train_epoch = 2000
    Total_interf_train = np.zeros([number_train_epoch, n_BS])

    for train_index in range(number_train_epoch):
        p_train = np.random.rand(n_BS) * P_MAX
        channel_index = np.random.randint(0, channel_map_length, 1)[0]

        total_train_inter_vec, total_train_inter_mat = _compute_interference_(UE_sched,
                                                                              p_train,
                                                                              BS_UE_distance,
                                                                              Antenna_gain_BS,
                                                                              Antenna_gain_UE,
                                                                              Fading_mat[channel_index, :, :],
                                                                              pathloss)
        Total_interf_train[train_index, :] = total_train_inter_vec

    """begin trial loop"""
    for trial_idx in range(n_trials):
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

        "Queue Initialization"
        # max_throughput = T_f * W * np.log2(1 + g_max * p_max)
        throughput_initial_value = 0
        power_initial_value = 0
        Z_i = power_initial_value * np.zeros(n_BS)
        H_ji = throughput_initial_value * np.zeros(n_BS)
        X_ji = np.zeros(n_BS)

        """//// begin slot loop ///"""
        # slot_idx = 0
        # Q_table_frames = np.zeros(shape=(n_frames, P_q, I_q, n_BS))  # initial zeros
        # Q_table_frames = np.ones(shape=(n_frames, P_q, I_q, n_BS))  # initial zeros
        Q_table = np.ones(shape=(P_q, I_q, n_BS))  # initial ones
        for frame_idx in range(n_frames):
            # if frame_idx % 100 == 0:
            #     learning_rate = max(learning_rate - 0.1, 0.1)
            power_profile_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            # Q_table = Q_table_frames[frame_idx, :, :, :]
            if frame_idx == 0:
                # Q_table = Q_table_frames[frame_idx, :, :, :]
                power_profile_frame[:, 0] = Power_profile[:, frame_idx]
            else:
                # Q_table = Q_table_frames[frame_idx-1, :, :, :]
                power_profile_frame[:, 0] = Power_profile[:, frame_idx - 1]

            reward_profile_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            Interf_profile_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            Interf_profile_frame_ = np.zeros([n_BS, n_slots], dtype=np.float32)
            Indiv_interf_profile_frame = np.zeros([n_BS, n_BS, n_slots], dtype=np.float32)
            Indiv_interf_profile_frame_ = np.zeros([n_BS, n_BS, n_slots], dtype=np.float32)
            U_i_frame = np.zeros([n_BS, n_slots], dtype=np.float32)
            X_ji_frame = np.zeros(n_BS, dtype=np.float32)

            grady_attempt = [0 for i in range(n_BS)]
            for slot_idx in range(n_slots):
                Current_state = np.zeros(n_BS)
                Selection_action = np.zeros(n_BS)
                Next_state = np.zeros(n_BS)

                if slot_idx == 0:
                    previous_power = power_profile_frame[:, slot_idx]
                else:
                    previous_power = power_profile_frame[:, slot_idx - 1]
                # print(slot_idx, 'previous power', previous_power)
                """compute interf at current slot (channel+power) """
                interf_vec, Interf_mat = _compute_interference_(UE_sched,
                                                                previous_power,
                                                                BS_UE_distance,
                                                                Antenna_gain_BS,
                                                                Antenna_gain_UE,
                                                                Fading_mat[frame_idx*n_slots + slot_idx-1, :, :],
                                                                pathloss)

                Interf_profile_frame[:, slot_idx] = interf_vec
                Indiv_interf_profile_frame[:, :, slot_idx] = Interf_mat  # component interf g_ki(t)p_ki(t)

                """Get instant power according to Q table"""
                for bs in range(n_BS):
                    Current_state[bs] = find_state_percentile(I_q=I_q, I_quan=I_quan, x=interf_vec[bs],
                                                              A=Total_interf_train[:, bs])
                    tmp_epsilon = np.random.rand()
                    if tmp_epsilon <= epsilon:
                        grady_attempt[bs] += 1
                        Selection_action[bs] = np.random.randint(0, P_q, 1)
                    else:
                        arr = Q_table[:, int(Current_state[bs]), bs]
                        max_value = max(Q_table[:, int(Current_state[bs]), bs])
                        Selection_action[bs] = list(arr).index(max_value)

                    # print(bs, idx, Selection_action[bs], int(Selection_action[bs]))
                    power_profile_frame[bs, slot_idx] = P_quan[int(Selection_action[bs])]
                # print(slot_idx, 'selected power', power_profile_frame[:, slot_idx])

                """ compute decision point interf (at the beginning of next slot)"""
                interf_vec_, Interf_mat_ = _compute_interference_(UE_sched,
                                                                 power_profile_frame[:, slot_idx],
                                                                 BS_UE_distance,
                                                                 Antenna_gain_BS,
                                                                 Antenna_gain_UE,
                                                                 # Fading_mat[min(frame_idx+1, n_frames-1), :, :],
                                                                 Fading_mat[frame_idx*n_slots + slot_idx-1, :, :],
                                                                 pathloss)
                Interf_profile_frame_[:, slot_idx] = interf_vec_
                Indiv_interf_profile_frame_[:, :, slot_idx] = Interf_mat_  # component interf g_ki(t+1)p_ki(t)

                # print(slot_idx, interf_vec, interf_vec_)
                """reward function need to be replaced by lyapunov optimization framework (alpha/beta)"""
                for bs in range(n_BS):
                    """Select next state"""
                    Next_state[bs] = find_state_percentile(I_q=I_q, I_quan=I_quan, x=interf_vec_[bs],
                                                           A=Total_interf_train[:, bs])
                    """ compute reward (throughput/sec/Hz) """
                    ue_sch = UE_sched[bs]
                    ue_g = bs * n_UE_per_BS + ue_sch
                    SINR = power_profile_frame[bs, slot_idx] * antenna_gain_max_BS * antenna_gain_max_UE * \
                           _pathloss_(BS_UE_distance[ue_g][bs], pathloss) * \
                           Direct_channel[frame_idx*n_slots + slot_idx-1, ue_sch, bs] / (Interf_profile_frame[bs, slot_idx] + noise * W)
                    """Update alpha and beta according to current Lyapunov queue"""
                    alpha = H_ji[bs] * T_b  # T_b is (fixed) block time, need to be defined previously
                    beta = Z_i[bs] * T_b

                    X_ji_frame[bs] += T_unit * W * np.log2(1 + SINR)  # T_b and W need to be defined previously
                    reward_profile_frame[bs, slot_idx] = alpha * W * np.log2(1 + SINR) - beta * power_profile_frame[bs, slot_idx]

                    """Q table update"""
                    Q_max = max(Q_table[:, int(Next_state[bs]), bs])
                    tmp_diff = reward_profile_frame[bs, slot_idx] + discount_factor * Q_max - Q_table[
                        int(Selection_action[bs]), int(Current_state[bs]), bs]
                    Q_table[int(Selection_action[bs]), int(Current_state[bs]), bs] = Q_table[int(Selection_action[bs]), int(Current_state[bs]), bs] \
                                                                                     + learning_rate * tmp_diff

            # print(frame_idx, grady_attempt)
            # Q_table_frames[frame_idx, :, :, :] = Q_table
            Power_profile[:, frame_idx] = power_profile_frame[:, -1]
            Interf_profile[:, frame_idx] = Interf_profile[:, -1]
            Reward_profile[:, frame_idx] = reward_profile_frame[:, -1]
            U_i[:, frame_idx] = Utility_function(X_ji_frame, utility_power=0.6)
            X_ji = X_ji_frame
            # score_history.append(np.mean(Reward_profile[:, frame_idx]))
            score_history.append(np.mean(U_i[:, frame_idx]))

            """Solve Lyapunov optimization (Auxiliary Variables)"""
            x = cp.Variable(n_BS)
            # objective = cp.Minimize(-V * cp.sum(cp.log(x) / cp.log(2).value) + cp.sum(H_ji.T @ x))
            objective = cp.Minimize(-V * cp.sum(cp.sum(Utility_function(x, utility_power=0.6))) + cp.sum(H_ji.T @ x))
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
                    trial_idx, frame_idx, n_frames, score_history[frame_idx], np.mean(score_history[-100:])))
                print("--------------------------------------")

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
    Rwd_avg_BS = Reward_profile_trial.mean(axis=1)  # avg over BSs,size=(num_rep,n_slots)
    Utility_avg_BS = U_trail.mean(axis=1)  # avg over BSs,size=(num_rep,n_slots)

    Rwd_over_trials = np.zeros([n_trials, n_frames])
    Utility_over_trials = np.zeros([n_trials, n_frames])
    for i in range(n_trials):
        Rwd_over_trials[i, :] = _move_avg_(Rwd_avg_BS[i, :], window_size=window_size)
        for j in range(n_frames):
            Utility_over_trials[i, j] = sum(Utility_avg_BS[i, :j+1]) / len(Utility_avg_BS[i, :j+1])

    """save plot data, rows 0,1,2-> mean, max, min """
    Data = np.zeros([4, n_frames])

    # Data[0, :] = Rwd_over_trials.mean(axis=0)
    # Data[1, :] = Rwd_over_trials.max(axis=0)
    # Data[2, :] = Rwd_over_trials.min(axis=0)
    # Data[3, :] = Rwd_over_trials.std(axis=0)

    Data[0, :] = Utility_over_trials.mean(axis=0)
    Data[1, :] = Utility_over_trials.max(axis=0)
    Data[2, :] = Utility_over_trials.min(axis=0)
    Data[3, :] = Utility_over_trials.std(axis=0)

    return Data, Reward_profile_trial[-1], Power_profile_trial[-1], Power_avg_profile_trail[-1], H_ji_profile_trail[-1]  # last trial rewards, powers,size=(n_BS,n_slots)


def main(IsSmallNet=True, n_trials=1, window_size=200, n_frames=1000, n_slots=500, bandwidth=500e6):

     p_max_dBm = 39
     TrainMode = True
     LoadModel = not True  # train mode off if model is loaded
     min_noise = 0

     """Lyapunov related parameters"""
     V = 5000  # 3000 Example
     T_b = 0.5  # 1 block per frame
     T_f = 0.5
     T_unit = T_b / n_slots
     # T_unit = 1e-3
     # T_b = n_slots * T_unit
     # T_f = n_slots * T_unit
     W = 1
     # W = 20e3
     # W = 400e6
     P_avg_dBm = 38.13  # 33
     # P_avg_dBm = 35
     P_avg = 1e-3 * 10 ** (P_avg_dBm / 10)
     g_max = 1e12
     # g_max = 1e20
     # pathloss = ["dual_slope", 2, 2, 1]
     pathloss = ["dual_slope", 4, 4, 1]
     # pathloss = ["dual_slope", 6, 6, 1]

     """
     NOTE:
     loss are same for two range in this case, QL and GT performance are highly dependent on value of loss
     smaller loss leads to better performance of GT, even better that QL
     """

     # beamwidth = np.pi / 4
     beamwidth = np.pi / 6

     "Q-Learning parameters"
     P_q = 100  # 40
     I_q = 20  # 16
     learning_rate = 0.1
     discount_factor = 0.9
     number_train_epoch = 1000
     epsilon = 0.05

     """read simulation configurations"""
     with open('parameters.txt', 'r') as file:
         json_str = file.read()
     params = json.loads(json_str)  # Convert the JSON string to a dictionary
     Data_QL, Rwd_QL, Power_QL, Power_avg, H_ji = main_QL_w_Lyapunov(n_frames=n_frames, n_slots=n_slots, n_trials=n_trials,
                    test_channel=0, MSR=20, beamwidth=beamwidth, power_max_dBm=p_max_dBm, bandwidth=bandwidth,
                    UE_sched=0, discount_factor=discount_factor, pathloss=pathloss, SmallNet=True, window_size=window_size,
                    V=V, T_f=T_f, T_b=T_b, T_unit=T_unit, W=W, P_avg=P_avg, g_max=g_max, P_q=P_q, I_q=I_q,
                    learning_rate=learning_rate, number_train_epoch=number_train_epoch, epsilon=epsilon)

     time_finished = time.strftime("%H:%M:%S", time.localtime())
     np.save('npy_data/QL_data_{}_{}_{}_{}_{}.npy'.format(time_finished, P_q, I_q, learning_rate, number_train_epoch), Data_QL)

     MSR, BW, n_BS = params["MSR"], params["BW"], len(Power_QL[:, 0])
     label = "QL throughput."

     plt.plot(Data_QL[0], label=label, lw=0.8)
     plt.ylabel('Average throughput QL (bits/sec/Hz per BS)')
     plt.xlabel('Frames')
     # plt.ylim([min(Data_MADDPG[2]) - 0.5, max(Data_MADDPG[1]) + 0.5])
     plt.ylim([0, max(Data_QL[1]) + 0.1])
     plt.legend(loc='lower right')
     plt.grid()
     # plt.savefig('figure/MADDPG_fig/MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     fig, ax = plt.subplots(n_BS, 1, sharex=True)  # plot powers
     fig.suptitle(f"QL power over time")
     for bs in range(n_BS):
         ax[bs].plot(Power_QL[bs, -int(n_frames / 1):-1], label=f"BS {bs}", lw=0.8)
         ax[bs].set_title(f'BS {bs}', fontsize=8)
         ax[bs].set_ylim([0, dBm2Watt(p_max_dBm)])
         ax[bs].grid()
         if bs == n_BS - 1:
             ax[bs].set_xlabel('Slots')
     # plt.savefig('figure/Power_fig/Power_instant_MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     fig, ax = plt.subplots(n_BS, 1, sharex=True)  # plot powers
     fig.suptitle(f"QL H_ji over time")
     for bs in range(n_BS):
         ax[bs].plot(H_ji[bs, -int(n_frames / 1):-1], label=f"BS {bs}", lw=0.8)
         ax[bs].set_title(f'BS {bs}', fontsize=8)
         ax[bs].grid()
         if bs == n_BS - 1:
             ax[bs].set_xlabel('Frames')
     # plt.savefig('figure/Lyapunov_fig/H_ji_MADDPG_{}.pdf'.format(time_finished))
     plt.show()

     fig, ax = plt.subplots(n_BS, 1, sharex=True)
     fig.suptitle(f"QL P_avg over time")
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
         n_trials=1,
         n_frames=n_frames,
         n_slots=n_slots,
         bandwidth=400e6,
         window_size=250)
