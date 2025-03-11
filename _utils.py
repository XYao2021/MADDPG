import math
import numpy as np
import random as rand
from scipy.stats import nakagami


def dBm2Watt(p):
    r"""convert dBm to Watt"""
    return (10**(p/10.0))/1000.0

def gen_init_power(n_BS, mode="zero", predef_power=None, tanh_factor=0.999):
    r"""generate initial power (under Tanh actor)
    Input:
        mode: "zero", "full", or "predefined", string type
        predef_power: if mode=="predefined", then need specify initial power
            size=(n_BS), entry range [0,1]
        tanh_factor: Tanh clip factor
    Output:
        acts: initial actions (not converted to real power yet), entry range [-factor,factor]
     """
    if mode == "zero":
        acts = [np.array([-tanh_factor]) for _ in range(n_BS)]  # tanh: -factor-> p=0
    elif mode == "full":
        acts = [np.array([tanh_factor]) for _ in range(n_BS)]  # factor -> p_max
    elif mode == "predefined":
        # acts = [np.array([predef_power[k]]) for k in range(n_BS)]
        acts = [np.array([unit_map_inverse(predef_power[k], factor=tanh_factor)]) for k in range(n_BS)]

    return acts


def point_in_circle(x, y, r_min, r_max, seed=0):
    r"""generate a random 2D point within two circles with radius r_min and r_max
        for each input data point
    INPUT:
        x: x axis of input, type: np array, size =(n,)
        seed: radom seed
    OUTPUT:
        x_: generated points, size=(n,)
        y_
        """
    x = np.array(x)
    y = np.array(y)
    n_data = x.shape[0]
    x_ = np.zeros(n_data)
    y_ = np.zeros(n_data)
    np.random.seed(seed)
    radius_all = np.random.uniform(r_min, r_max, size=n_data)   #distances from centers
    angles_all = np.random.uniform(0, 2*np.pi, size=n_data)    #angles relative to centers
    for i in range(n_data):
        x_[i] = x[i] + np.cos(angles_all[i]) * radius_all[i]
        y_[i] = y[i] + np.sin(angles_all[i]) * radius_all[i]
    
    return x_, y_
    

def noise_sigma(bwidth=100*1e6):
    r""" Compute total noise over bandwidth
        INPUT:
            bwidth: bandwidth (Hz), default to 100 MHz
        OUTPUT: 
            sigma: noise power
    """
    k_B = 1.38e-23   #Boltzman constant
    NR = 1.5     #noise figure (dB)
    T_0 = 290  # temperature (K)
    
    return 10*math.log10(k_B * T_0 * 1000) + NR + 10 * math.log10(bwidth)


def FP(Composite_channel, sched_UEs, power_max, noise, n_iters, init, weights, 
       p_init_factor=0.5, if_gamma_auto=False, gamma_init=1e6):
    """FP with properly-initialized gammas
        INPUT:
            Composite_channel: equivalent channels, size=(n_UE,n_BS), entry at (ue,bs)
                is the channel from BS bs to UE ue (global UE index)
            sched_UEs: sched UE (local index) of each BS, size=(n_BS,) 
            n_iters: number of inner iterations for FP
            init: "constant" or "random", power init method, type: str
            p_init_factor: init power=p_init_factor*p_max
            if_gamma_auto: True if adaptive gamma init is used, ow specify 
                use gamma_init as init values of gamma
            gamma_init: init values of gamma (SINR), will not be used if if_gamma_auto 
                is True (default to large value 1e6 to avoid FP oscillation)
        OUTPUT:
            P: powers of the last innter iteration,size=(n_BS,)
            Power_profile:powers at all innter iters, size=(n_BS,n_iters)
            Gamma_profile: gamma(SINR) at all innter iters, size=(n_BS,n_iters)
    """
    Composite_channel = np.array(Composite_channel)   #convert to arrays
    n_BS = Composite_channel.shape[1]
    n_UE = Composite_channel.shape[0]
    n_UE_per_BS = int(n_UE/n_BS)
    Gamma_profile = np.zeros([n_BS, n_iters])  #SINR
    Power_profile = np.zeros([n_BS, n_iters])  #record powers in each iter
    """init P, gamma, Y"""
    c = np.random.uniform(0.1, 0.8) if init == "random" else p_init_factor
    P = c * power_max * np.ones(n_BS)
    #init gamma
    SINR_max = np.zeros(n_BS)   #max possible SINR of each BS
    for bs in range(n_BS):
        SINR_max[bs] = Composite_channel[bs*n_UE_per_BS+sched_UEs[bs], bs]*power_max/noise
    gamma = SINR_max if if_gamma_auto else gamma_init * np.ones(n_BS)
    Y = 1 * np.ones(n_BS)  # Y is determined by P, gamma, Y0 does not matter
    """ === begin iteration ===="""
    for iter_idx in range(n_iters):
        """ update Y """
        for i in range(n_BS):
            ue_g = i*n_UE_per_BS + sched_UEs[i]
            numer = np.sqrt(weights[i]*(1.0 + gamma[i])*Composite_channel[ue_g, i]*P[i])
            temp = [Composite_channel[ue_g, j]*P[j] for j in range(n_BS)]
            denom = sum(temp) + noise 
            Y[i] = numer/denom
        """ update gamma (SINR) """
        for i in range(n_BS):
            ue_g = i*n_UE_per_BS + sched_UEs[i]
            numer = Composite_channel[ue_g, i] * P[i]
            temp = [Composite_channel[ue_g, j] * P[j] for j in range(n_BS) if j != i]
            denom = sum(temp) + noise
            gamma[i] = numer/denom
        Gamma_profile[:, iter_idx] = gamma
        """ update P """
        for i in range(n_BS):
            ue_g = i*n_UE_per_BS + sched_UEs[i]
            numer = Y[i]**2*weights[i]*(1.0 + gamma[i])*Composite_channel[ue_g, i]
            temp = [Y[j]**2*Composite_channel[j*n_UE_per_BS+sched_UEs[j], i] for j in range(n_BS)]
            P[i] = min(power_max, numer/sum(temp)**2) 
        Power_profile[:, iter_idx] = P
    return P, Power_profile, Gamma_profile


def WMMSE(Composite_channel, sched_UEs, power_max, noise, n_iters, init, weights):
    """WMMSE power allocation
        INPUT:
            Composite_channel: equivalent channels, size=(n_UE,n_BS), entry at (ue,bs)
                is the channel from BS bs to UE ue (global UE index)
            sched_UEs: sched UE (local index) of each BS, size=(n_BS,) 
            power_max: max power
            noise: noise power
            n_iters: number of inner iterations for WMMSE
            init: "constant" or "random", power init method, type: str
            weights: weights of each BS's throughput, default to one
        OUTPUT:
            power_vec: power of the last WMMSE inner iteration,size=(n_BS,)
            Power_mat: powers of all inner iterarions,size=(n_BS,n_iters)
            """
    Composite_channel = np.array(Composite_channel)
    n_BS = Composite_channel.shape[1]
    n_UE = Composite_channel.shape[0]
    n_UE_per_BS = int(n_UE/n_BS)
    V_mat = np.zeros([n_BS, n_iters])          # record V vals for all BSs in all iters
    """ === initialize V,U,W === """
    if init == "random":
        V = np.random.rand()*np.sqrt(power_max)*np.ones(n_BS)         #random initialization
    elif init == "constant":
        V = 0.5*np.sqrt(power_max)*np.ones(n_BS)  #V_init=0.5*sqrt(P_max)
    """ initialize U """
    U = np.zeros(n_BS)
    for bs in range(n_BS):
        ue_sched_g = bs*n_UE_per_BS + sched_UEs[bs]              # sched UE global index
        numer = np.sqrt(Composite_channel[ue_sched_g, bs])*V[bs]      # numerator
        vec = [Composite_channel[ue_sched_g, j] * V[j]**2 for j in range(n_BS)]
        denom = sum(vec) + noise
        U[bs] = numer/denom
    """ initialize W """
    W = np.zeros(n_BS)
    for bs in range(n_BS):
        ue_sched_g = bs*n_UE_per_BS + sched_UEs[bs] 
        temp = 1.0 - U[bs]*np.sqrt(Composite_channel[ue_sched_g, bs])*V[bs]
        W[bs] = 1.0 / max(1e-15, temp)     #for numerical stability
    """ === iteratively compute V,U,W ==="""
    for i in range(n_iters):
        """ -- update V --"""
        for bs in range(n_BS):
            ue_sched_g = bs*n_UE_per_BS + sched_UEs[bs] 
            v_numer = weights[bs]*W[bs]*U[bs]*np.sqrt(Composite_channel[ue_sched_g, bs])
            vec = [weights[j] * W[j]*U[j]**2 * \
                Composite_channel[j*n_UE_per_BS+sched_UEs[j], bs] for j in range(n_BS)]
            v_denom = sum(vec)
            V[bs] = np.clip(v_numer/v_denom, 0.0, np.sqrt(power_max))
        V_mat[:, i] = V
        """ -- update U --"""
        for bs in range(n_BS):
            ue_sched_g = bs*n_UE_per_BS + sched_UEs[bs]
            u_numer = np.sqrt(Composite_channel[ue_sched_g, bs]) * V[bs]
            vec = [Composite_channel[ue_sched_g, j]*V[j]**2 for j in range(n_BS)]
            u_denom = sum(vec) + noise
            U[bs] = u_numer/u_denom
        """ -- update W --"""
        for bs in range(n_BS):
            ue_sched_g = bs*n_UE_per_BS + sched_UEs[bs]
            temp = 1.0 - U[bs]*np.sqrt(Composite_channel[ue_sched_g, bs])*V[bs]
            W[bs] = 1.0/max(1e-15, temp)
    power_vec = np.square(V)
    Power_mat = np.square(V_mat)    
    return power_vec, Power_mat


def temporal_Nakagami(n_trials, n_slots, n_BS, n_UE, m=20, Omega=1.0, rho=0.3):
    """ Generate time-correlated Nakagami rvs with fixed m & Omega,
        and controllable correlation 
        coefficient (rho) 
        INPUT:
            Omega: Omega:=E[h^2]=1 by default & fixed
            m: m=E[h^2]^2/Var(h^2) =1/Var(h^2), type: integer/.
            rho: rho = lambda ^ 4, correlation coefficient
    """
    Lambda = np.power(rho, 1/4)
    Out = np.zeros([n_trials, n_slots, n_UE, n_BS])
    m = int(m)
    noise_init = np.ones(m)   #ensures h^2 has mean 1
    for i in range(n_trials):
        print(f"trial:{i}")
        for j in range(n_slots):
            for ue in range(n_UE):
                for bs in range(n_BS):
                    new_noise = np.random.normal(size=[m])
                    Out[i, j, ue, bs] = np.sqrt(Omega/m)*np.linalg.norm(
                                np.sqrt(1-Lambda**2)*new_noise+Lambda*noise_init)
    return Out

def RayleighFading(input_size, Omega):
    """ Generate Rayleigh fading  
    INPUT:
        Omega: :=E[X^2]
    """
    Fading = np.random.rayleigh(scale=math.sqrt(Omega/2), size=input_size)
    return Fading
    

def zero_padding(data_vec, idx_vec):
    """ INPUT:
            data_vec: 1D array of scalars
            idx_vec: 1D array of integers or None
            * |data_vec| >= |idx_vec|
        OUTPUT:
            out_vec: out_vec[i] = data_vec[i] if idx_vec[i]!=None, ow out_vec[i] =0.0
                size=(|idx_vec|)
    """
    length = min(len(data_vec), len(idx_vec))
    out_vec = np.zeros(length)
    for i in range(length):
        if idx_vec[i] is not None:
            out_vec[i] = data_vec[idx_vec[i]]
        else:
            0.0
    return out_vec
    
def extract_local_obs(local_obs_all, neighbor_set):
    """ Extract a subset of local observations (each type: 1D array) for each 
        agent i from all its neighbors (including i)
        INPUT:
            local_obs_all: list of local obs (1D array) of all agents, list_size=n_agents, 
                entry_array_size = (actor_dim)
            neighbor_set: {0:[1,2], 1:[2,3],....}, type: dict, dict_size=n_agents, 
                entry_list_size= neighbor_set_size
        OUTPUT:
            local_obs_subset: {0: [array(obs0),array(obs1),array(obs2)],....}, type: dict,
                dict value type: list, list entry type: array
                returns a list of local obs of agent i and its nieghbors
    """
    actor_dim = local_obs_all[0].shape
    local_obs_subset = dict({})
    for i in neighbor_set:
        local_obs_subset[i] = [local_obs_all[i]] +\
                [local_obs_all[j] if j is not None else np.zeros(actor_dim)
                 for j in neighbor_set[i]]  #zero padding if None
                            
    return local_obs_subset
        
 
def unit_map(x, factor=1.0):
    """map [-factor,factor] to [0,1]"""
    return (np.array(x)+factor)/(2*factor)


def unit_map_inverse(x, factor=1.0):
    r"""map [0,1] to [-factor, factor]"""
    return (np.array(x) - 0.5) * 2 * factor

def Utility_function(x, utility_power):
    return x**utility_power

def Utility_function_log(x):
    return np.log(x)

def _compute_distance_(Pos_BS, Pos_UE, height_BS, height_UE):
    """Computes the distance between BSs and UEs
        INPUT:
            Pos_BS: size: number_BS*2, type: numpy arrary
            Pos_UE: size: number_UE*2, type: numpy arrary
            height_BS: scalar
            height_UE: scalar
        OUTPUT:
            Dist_mat: size=(number_UE, number_BS)
    """
    Pos_BS = np.array(Pos_BS)        # convert to arrays
    Pos_UE = np.array(Pos_UE)
    num_BS = Pos_BS.shape[0]
    num_UE = Pos_UE.shape[0]
    Dist_mat = np.zeros([num_UE, num_BS])
    for ue in range(num_UE):
        for bs in range(num_BS):
            # planar distance
            temp = math.sqrt(sum((Pos_UE[ue, :] - Pos_BS[bs, :])**2))  # Square first, then take the summation?
            # space distance
            Dist_mat[ue][bs] = math.sqrt(temp**2 + (height_BS - height_UE)**2)
    return Dist_mat
            

def _pathloss_(dist, Pathloss):
    """Computes path loss
        INPUT:
            dist: scalar or array, list or array
            Pathloss=[model, exp_0, exp_1, critical_dist], type = list 
        OUTPUT:
            ploss: pathloss, scalar
    """
    # model = Pathloss[0]
    # exp_0 = Pathloss[1]
    # exp_1 = Pathloss[2]
    # critical_dist = Pathloss[3]
    # dist = float(dist)
    # if model == 'dual_slope':
    #     if dist <= critical_dist:
    #         ploss = 1/dist**(exp_0)
    #     else:
    #         ploss = critical_dist**(exp_1 - exp_0) / dist**(exp_1)
    # elif model == 'uniform':
    #     ploss = dist ** (-exp_0)
    # return ploss
    return 1/(dist ** 4)


def JakesFading_Gaussian(n_trials, n_iters, n_UE, n_BS, noise_init=1.0, rho=0.9, scale=1, rand_seed=42):
    """ Generate Jake's fading h(t)=rho*scale*h(t-1)+sqrt(1-rho^2)*scale *n(t)
        n(t) ~ N(0,1), std Gaussian
        INPUT:
            noise_init: h(0), scalar
            rho: channel correlation, in [0,1]
            scale: control variance of |h|^2
            rand_seed: random seed
        OUTPUT:
            Fading: size =(n_trials,n_iters,n_UE,n_BS)
    """
    np.random.seed(rand_seed)
    Fading = np.zeros([n_trials, n_iters, n_UE, n_BS])
    for trial_idx in range(n_trials):
        Fading[trial_idx, 0, :, :] = noise_init*np.ones([n_UE, n_BS])  #first iteration/slot channels
        for iter_idx in range(1, n_iters):
            Fading[trial_idx, iter_idx, :, :] = rho*scale*Fading[trial_idx, iter_idx-1, :, :]\
                + np.sqrt(1.0-rho**2)*scale*np.random.normal(size=[n_UE, n_BS])
        print(f'Trail {trial_idx} finished.')
    return Fading


def JakesFading_Nakagami(n_trials, n_iters, n_UE, n_BS, noise_init=1.0, nu=2, rho=0.9, scale=1):
    """ Generate Jake's fading h(t)=rho*scale*h(t-1)+sqrt(1-rho^2)*scale*n(t)
        n(t): Nakagami with E[|h|^2]=1, var(|h|^2) = 1/nu  scale=Omega, nu=m
        INPUT:
            noise_init: h(0), scalar
            rho: channel correlation, in [0,1]
            scale: control variance of |h|^2
        OUTPUT:
            Fading: size =(n_trials,n_iters,n_UE,n_BS)
    """
    Fading = np.zeros([n_trials, n_iters, n_UE, n_BS])
    for trial_idx in range(n_trials):
        Fading[trial_idx, 0, :, :] = noise_init*np.ones([n_UE, n_BS])
        for iter_idx in range(1, n_iters):
            Fading[trial_idx, iter_idx, :, :] = rho*scale*Fading[trial_idx, iter_idx-1, :, :]\
                + np.sqrt(1.0-rho**2)*scale*nakagami.rvs(nu=nu, size=[n_UE, n_BS])
        print(f"Trail {trial_idx} finished.")
    return Fading

def Time_correlated_Rayleigh(n_trials, n_iters, n_UE, n_BS, noise_init=1.0, rho=0.9, scale=1, Omega=1):
    Fading = np.zeros([n_trials, n_iters, n_UE, n_BS])
    for trial_idx in range(n_trials):
        Fading[trial_idx, 0, :, :] = noise_init*np.ones([n_UE, n_BS])
        for iter_idx in range(1, n_iters):
            Fading[trial_idx, iter_idx, :, :] = rho*scale*Fading[trial_idx, iter_idx-1, :, :]\
                + np.sqrt(1.0-rho**2)*scale*RayleighFading(input_size=[n_UE, n_BS], Omega=Omega)
        print(f"Trail {trial_idx} finished.")
    return Fading

def IID_fading_Nakagami(n_trials, n_iters, n_UE, n_BS, model, nu): 
    """Generate small-scale fNakagami fading h_ji 
        INPUT:
            n_trials: number of trials, each having iid fading
            n_iters: number of iterations, each having iid fading
            model: fading distribution, 'nakagami' 
            nu: = E[|h|^2]^2/Var(|h|^2) = 1/Var(|h|^2), E[|h^2|] default to 1
        OUTPUT:
            Fading_mat: size =(n_trials, n_iters, n_UE, n_BS)
    """
    if model == 'nakagami': 
        Fading_mat = nakagami.rvs(nu, size=[n_trials, n_iters, n_UE, n_BS]) 
    else:
        Fading_mat = 'None'
        raise Exception("Wrong fading model!")
    return Fading_mat      # np array


def _unit_vector_(vector):
    """Normalize to unit vector """
    return vector / np.linalg.norm(vector)


def _get_angle_(vector): 
    """Returns the angle between 'vector' and (1,0)
        INPUT:
            vector: np arrary [x,y]
        OUTPUT:
            angle in [0, 2*pi]
    """
    v_u = _unit_vector_(vector)
    v_ref = np.array([1., 0.])
    angle = np.arccos(np.clip(np.dot(v_u, v_ref), -1.0, 1.0)) 
    if v_u[1] < 0:    
        angle = 2 * np.pi - angle        # arccos() returns output in [0, pi] 
    return angle       # output in [0, 2*pi)


def _beam_align_BS_(BS_position, UE_position, Scheduled_UE, beamwidth):
    """Align BS beams towards scheduled UEs
        INPUT:
            BS_position: size = (num_BS, 2)
            UE_position: size = (num_UE, 2)
            Scheduled_UE: (num_BS, )
            beamwidth: BS beamwidth
        OUTPUT:
            Beam_direction: size=(num_BS,2), i-th row=[beam_start_BSi,beam_end_BSi]
    """
    BS_position = np.array(BS_position)   # convert to array
    UE_position = np.array(UE_position)
    num_BS = BS_position.shape[0]
    num_UE = UE_position.shape[0]
    num_UE_per_BS = int(num_UE / num_BS)
    Beam_direction = np.zeros([num_BS, 2])
    for bs in range(num_BS):
        ue_g = bs * num_UE_per_BS + Scheduled_UE[bs]       # index starts from 0    
        angle = _get_angle_(UE_position[ue_g, :] - BS_position[bs, :])
        Beam_direction[bs, :] = [angle - beamwidth / 2, angle + beamwidth / 2]
        # note that left boundary < 0 or right boundary > 2*pi can happen
    return Beam_direction 


def _beam_align_UE_(BS_position, UE_position, beamwidth):
    """Align UE beams towards associated BSs
        INPUT:
            BS_position: num_BS*2
            UE_position: num_UE*2
            beamwidth: UE beamwidth
        OUTPUT:
            Beam_direction: i-th row = [beam_start_UEi, beam_end_UEi]
    """
    BS_position = np.array(BS_position)
    UE_position = np.array(UE_position)
    num_BS = BS_position.shape[0]
    num_UE = UE_position.shape[0]
    num_UE_per_BS = int(num_UE / num_BS)
    Beam_direction = np.zeros([num_UE, 2])
    for ue in range(num_UE):
        bs = math.floor(ue / num_UE_per_BS)     # assocaited BS of UE ue
        angle = _get_angle_(BS_position[bs, :] - UE_position[ue, :])
        Beam_direction[ue, :] = [angle - beamwidth / 2, angle + beamwidth / 2]
        # note that left boundary < 0 or right boundary > 2*pi can happen
    return Beam_direction    


def _antenna_gain_BS_(BS_position, UE_position, Beam_direction, antenna_MSR, beamwidth):
    """Calculate BS antenna gain toward all UEs
        INPUT:
            BS_position
            UE_position 
            Beam_direction: BS beam direction matrix, size=(num_BS,2)
            antenna_MSR: BS MSR in dB
            beamwidth: BS beamwidth
        OUTPUT:
            Antenna_gain_mat: BS antenna gains towards all UEs, size=(num_UE,num_BS)
    """
    antenna_gain_min = 1/(beamwidth*10**(antenna_MSR/10) + 2*np.pi - beamwidth) 
    antenna_gain_max = 10**(antenna_MSR/10) * antenna_gain_min
    BS_position = np.array(BS_position)   
    UE_position = np.array(UE_position)   
    Beam_direction = np.array(Beam_direction)  # convert to arrays
    num_BS = BS_position.shape[0]
    num_UE = UE_position.shape[0]
    Antenna_gain_mat = np.zeros([num_UE, num_BS])
    if beamwidth == 2 * np.pi:
        Antenna_gain_mat = antenna_gain_min * np.ones([num_UE, num_BS])      #omnidirectional
    else:
        antenna_cases = []                     # determine 3 diff. beam coverage cases
        flag = None
        for bs in range(num_BS):
            if (Beam_direction[bs][0] > 0.) and (Beam_direction[bs][1] < 2 * np.pi):
                flag = 'case 0'
            elif Beam_direction[bs][0] <= 0.:
                flag = 'case 1' 
            elif Beam_direction[bs][1] >= 2 * np.pi:
                flag = 'case 2'
            antenna_cases.append(flag) 
            
        for bs in range(num_BS):
            for ue in range(num_UE):
                angle = _get_angle_(UE_position[ue, :]-BS_position[bs, :])  #in [0,2*pi)
                if antenna_cases[bs] == 'case 0':
                    if Beam_direction[bs][0] <= angle <= Beam_direction[bs][1]:
                        Antenna_gain_mat[ue][bs] = antenna_gain_max
                    else:
                        Antenna_gain_mat[ue][bs] = antenna_gain_min       
                if antenna_cases[bs] == 'case 1':
                    if (0.0 <= angle <= Beam_direction[bs][1]) or (2*np.pi+Beam_direction[bs][0]
                                                                 <= angle <= 2*np.pi):
                        Antenna_gain_mat[ue][bs] = antenna_gain_max
                    else:
                        Antenna_gain_mat[ue][bs] = antenna_gain_min
                if antenna_cases[bs] == 'case 2':
                    if (0.0 <= angle <= Beam_direction[bs][1]-2*np.pi) or (Beam_direction[bs][0]
                                                                  <= angle <= 2*np.pi):
                         Antenna_gain_mat[ue][bs] = antenna_gain_max
                    else:
                         Antenna_gain_mat[ue][bs] = antenna_gain_min
    
    return Antenna_gain_mat
                    

def _antenna_gain_UE_(BS_position, UE_position, Beam_direction, antenna_MSR, beamwidth):
    """Calculate UE antenna gains to all BSs
        INPUT:
            BS_position
            UE_position 
            Beam_direction: UE beam direction matrix, size = num_UE*2
            antenna_MSR: UE MSR in dB
            beamwidth: UE beamwidth
        OUTPUT:
            Antenna_gain_mat: UE antenna gains towards all UEs, num_UE*num_BS
    """
    antenna_gain_min = 1 / (beamwidth*10**(antenna_MSR / 10) + 2 * np.pi - beamwidth) 
    antenna_gain_max = 10**(antenna_MSR / 10) * antenna_gain_min
    BS_position = np.array(BS_position)   
    UE_position = np.array(UE_position)   
    Beam_direction = np.array(Beam_direction)  # convert to array
    num_BS = BS_position.shape[0]
    num_UE = UE_position.shape[0]
    Antenna_gain_mat = np.zeros([num_UE, num_BS])
    if beamwidth == 2 * np.pi:
        Antenna_gain_mat = antenna_gain_min * np.ones([num_UE, num_BS])      #omnidirectional
    else:
        antenna_cases = []      # three diff. beam coverage cases
        flag = None
        for ue in range(num_UE):
            if (Beam_direction[ue][0] > 0.) and (Beam_direction[ue][1] < 2 * np.pi):
                flag = 'case 0'
            elif Beam_direction[ue][0] <= 0.:
                flag = 'case 1' 
            elif Beam_direction[ue][1] >= 2 * np.pi:
                flag = 'case 2'
            antenna_cases.append(flag) 
        for ue in range(num_UE):
            for bs in range(num_BS): 
                angle = _get_angle_(BS_position[bs, :] - UE_position[ue, :])
                if antenna_cases[ue] == 'case 0': 
                    if Beam_direction[ue][0] <= angle <= Beam_direction[ue][1]:
                        Antenna_gain_mat[ue][bs] = antenna_gain_max
                    else:
                        Antenna_gain_mat[ue][bs] = antenna_gain_min
                if antenna_cases[ue] == 'case 1':
                    if (0. <= angle <= Beam_direction[ue][1]) or (2 * np.pi + Beam_direction[ue][0] 
                                                                 <= angle <= 2*np.pi):
                        Antenna_gain_mat[ue][bs] = antenna_gain_max
                    else:
                        Antenna_gain_mat[ue][bs] = antenna_gain_min
                if antenna_cases[ue] == 'case 2':
                    if (0. <= angle <= Beam_direction[ue][1] - 2*np.pi) or (Beam_direction[ue][0]
                                                                  <= angle <= 2*np.pi):
                         Antenna_gain_mat[ue][bs] = antenna_gain_max
                    else:
                         Antenna_gain_mat[ue][bs] = antenna_gain_min

    return Antenna_gain_mat
    
    
def _compute_interference_(Scheduled_UE, Power_profile, BS_UE_distance, Antenna_gain_BS,  
                           Antenna_gain_UE, Fading_mat, Pathloss):
    """Compute Rx interference at the scheduled UEs of all BSs
        INPUT:
            Scheduled_UE: size=(num_BS,)
            Power_profile: powers of BSs, size=(num_BS,)
            BS_UE_distance: size = (num_UE,num_BS)
            Antenna_gain_BS: BS antenna gain after alignment, size=(num_UE, num_BS)
            Antenna_gain_UE: UE antenna gain after alignment, size=(num_UE, num_BS)
            Fading_mat: |h|^2, size=(num_UE, num_BS)
            Pathloss: list, [model='nakagami', exp_0, exp_1, d_C]
        OUTPUT:
            Intfer_vec: total Rx interf at each BS, size=(num_BS,)
            Inter_mat: individual interf from each BS,size=(num_BS,num_BS), 
                entry (i,j) is interf of BS j to Scheduled_UE[i] of BS i
    """
    BS_UE_distance = np.array(BS_UE_distance)  # convert to arrays
    Fading_mat = np.array(Fading_mat)
    num_UE, num_BS = BS_UE_distance.shape
    num_UE_per_BS = int(num_UE/num_BS)
    Interf_vec = np.zeros(num_BS)            # total interference at sched UEs
    Interf_mat = np.zeros([num_BS, num_BS])    # indiv. interf. of each BS to each sched UE
    for bs in range(num_BS): 
        ue_sched_g = bs*num_UE_per_BS + Scheduled_UE[bs]    # sched UE global index 
        total_interf = 0.0            # total interference at BS bs's sched UE
        for bs_ in [x for x in range(num_BS) if x != bs]:
            temp = Antenna_gain_BS[ue_sched_g, bs_]*Antenna_gain_UE[ue_sched_g, bs_] * \
                _pathloss_(BS_UE_distance[ue_sched_g, bs_], Pathloss) * \
                    Fading_mat[ue_sched_g, bs_]*Power_profile[bs_]
            total_interf += temp
        Interf_vec[bs] = total_interf  
        """ individual interf from each BS bbss """
        for bbss in range(num_BS):
            ue_g = bs*num_UE_per_BS + Scheduled_UE[bs]       # sched UE of BS bs
            Interf_mat[bs, bbss] = Antenna_gain_BS[ue_g, bbss] * \
                                    Antenna_gain_UE[ue_g, bbss] * \
                                    _pathloss_(BS_UE_distance[ue_g, bbss], Pathloss) * \
                                        Fading_mat[ue_g, bbss]*Power_profile[bbss]
    return Interf_vec, Interf_mat

def _find_state_percentile_(interference, Interf_histogram, num_state):
    """Determine the state value index (used with interference state quantization)
        INPUT:
            interference: scalar
            Interf_histogram: vector containing all recorded interference values, make histogram4
            ** the size of this must be large enough for the percentile estimation to be precise  
            num_state: number of dicrete inteference states (note that index starts from 0)
        OUTPUT:
            state_value: discrete interf. state index
            Warning:
            The 'linear' mehtod does not always return the percentile as a element in Interf_histogram.
            It is an estimate
    """
    temp = 100 / int(num_state)
    Percentile_vec = []
    for i in [x+1 for x in range(num_state)]:
        # p_temp = np.percentile(Interf_histogram, temp*i, method ='linear')   # reports error on Mac
        p_temp = np.percentile(Interf_histogram, temp*i)    
        Percentile_vec.append(p_temp)
        
    Percentile_vec = [0] + Percentile_vec      # list concatenation
    for i in range(num_state):
        if (Percentile_vec[i] <= interference) and (interference < Percentile_vec[i+1]):
            state = i
            break
        elif interference >= Percentile_vec[num_state]:
            state = num_state - 1
            break
    return Percentile_vec, state
    # state values start from 0


def _action_select_eps_greedy_(Q_table, state, epsilon):
    """Epsiloin greedy action selection
        INPUT:
            Q_table: size = num_action*num_state
            state: in range(num_state)
            epsilon: exploration probability
        OUTPUT:
            action: chosen action (in range(num_action))
    """
    Q_table = np.array(Q_table)
    num_action = Q_table.shape[0]
    temp = rand.uniform(0., 1.)
    if temp <= epsilon:
        action = np.random.randint(num_action)           #random action
    else:
        action = np.argmax(Q_table[:, int(state)])       #ties break aribitrarily
    return int(action)


def _Q_table_ini_all_(num_BS, num_action, num_state, mode):
    """Initialize Q_tables of all BSs
        INPUT:
            num_BS:
            num_action:
            num_state:
            mode: 'zeros', 'ones', 'random'
        OUTPUT:
            Q_table: size = num_action*num_state
    """
    num_BS = int(num_BS)
    num_action = int(num_action)   # convert to int
    num_state = int(num_state)
    size = [num_BS, num_action, num_state]
    if mode == 'zeros':
        Q_table = np.zeros(size)
    elif mode == 'ones':
        Q_table = np.ones(size)
    elif mode == 'random':
        Q_table = np.random.uniform(0, 1.0, size)      # uniform distribution in [0,1)
    return Q_table
        

def _simulate_phase_(num_slot_sim, Scheduled_UE, P_quantized, BS_UE_distance,
                     Antenna_gain_BS, Antenna_gain_UE, Fading_mat, Pathloss):
    """Find empirical interference distributiuon (w. randomly chosen powers)
    INPUT:
        num_slot_sim: no. of simulated slots
        Scheduled_UE: 
        P_quantized: quantized powers, size = num_action
        BS_UE_distance:
        Antenna_gain_BS: size = num_UE*num_BS
        Antenna_gain_UE: size = num_UE*num_BS
        Fading_mat: size = (num_slot, num_UE, num_BS), note num_slot_sim <= num_slot 
        Pathloss: = ['dual_slop', alpha0, alpha1, critical distance]
    OUTPUT:
        Interf_mat: size = num_BS*num_slot_sim
    """
    BS_UE_distance = np.array(BS_UE_distance)
    num_BS = BS_UE_distance.shape[1]
    Interf_mat = np.empty([num_BS, num_slot_sim])    # record interference
    P_mat = np.empty([num_BS, num_slot_sim])         # generate randomly chosen powers
    for i in range(num_BS):
        for j in range(num_slot_sim):
            P_mat[i, j] = rand.choice(P_quantized)
    for i in range(num_slot_sim):
        Interf_mat[:, i], _ =\
            _compute_interference_(Scheduled_UE, P_mat[:, i], BS_UE_distance, Antenna_gain_BS, Antenna_gain_UE, Fading_mat[i, :, :], Pathloss)
    return Interf_mat 
    

def _action_to_power_(Selected_action, P_quantized):
    """Convert action indices to actual discrete powers
        INPUT:
            Selected_action: action indices, size=(num_BS,)
            P_quantized: quantized powers
        OUTPUT:
            Selected_power: size=(num_BS,)
    """
    Selected_action = np.array(Selected_action)
    num_BS = Selected_action.shape[0]
    Selected_power = np.zeros(num_BS)
    for i in range(num_BS):
        Selected_power[i] = P_quantized[int(Selected_action[i])]
    return Selected_power


def _power_to_action_(Power_profile, P_quantized):
    """ Convert quantized powers to action indices 
        INPUT:
            Power_profile: powers, size = (num_BS,)
            P_quantized: discrete powers, size = (num_BS, )
        OUTPUT:
            Action_index: power level index, size = (num_BS,)
    """
    Power_profile = np.array(Power_profile)     
    num_BS = Power_profile.shape[0]
    P_quantized = list(P_quantized)      # convert to list
    Action_index = []
    for i in range(num_BS):
        Action_index.append(int(P_quantized.index(Power_profile[i])))
    return Action_index


def _move_avg_(in_vec, window_size):
    """Compute moving average with fixed window size
        INPUT:
            in_vec: size = (|in_vec|, )
            window_size: scalar, average over past window_size slots
        OUTPUT:
            out_vec: size = (|in_sec|, )
    """
    length = len(in_vec)
    in_vec = np.array(in_vec)    # convert to np array
    out_vec = np.zeros(length)
    for i in range(length):
        if i <= window_size - 1:
            out_vec[i] = np.mean(in_vec[0: i+1])
        else:
            out_vec[i] = np.mean(in_vec[i - window_size + 1: i+1])
    return out_vec        


def _composite_channel_(BS_UE_distance, Antenna_gain_BS, Antenna_gain_UE,
                        Fading_mat, Pathloss): 
    """Compute composite channel gains (=|h|^2 * antenna_gain * path_loss)
        INPUT:
            BS_UE_distance: (i,j): dist. from BS j to UE i, size= (num_UE, num_BS)
            Antenna_gain_BS: BS antenna gain after alignment, size = (num_UE, num_BS)
            Antenna_gain_UE: UE antenna gain after alignment, size = (num_UE, num_BS)
            Fading_mat: |h|^2, small-scale fading, size =(num_UE, num_BS)
        OUTPUT:
            Comp_channel:composite channel gain from each UE to each BS,size=(num_UE,num_BS)
                entry at (ue, bs) is the channel from BS bs to UE ue
    """                         
    BS_UE_distance = np.array(BS_UE_distance)
    num_BS = BS_UE_distance.shape[1]
    num_UE = BS_UE_distance.shape[0]
    Comp_channel = np.zeros([num_UE, num_BS])
    for bs in range(num_BS):
        for ue in range(num_UE):
            Comp_channel[ue, bs] = Antenna_gain_BS[ue, bs]*Antenna_gain_UE[ue, bs] * \
                _pathloss_(BS_UE_distance[ue, bs], Pathloss)*Fading_mat[ue, bs]
    return Comp_channel
                

def _scale_data_(data, max_val, min_val):
    """Normalize data to within range [0,1]"""
    return (data - min_val)/(max_val - min_val)


def move_avg(in_vec, window_size=100):
    """Moving average with fixed window size
        INPUT:
            in_vec: size = (|in_vec|, )
            window_size: scalar, average over past window_size slots
        OUTPUT:
            out_vec: size = (|in_sec|, )
    """
    length = len(in_vec)
    in_vec = np.array(in_vec)    # convert to np array
    out_vec = np.zeros(length)
    for i in range(length):
        if i <= window_size - 1:
            out_vec[i] = np.mean(in_vec[0: i+1])
        else:
            out_vec[i] = np.mean(in_vec[i - window_size + 1: i+1])
    return out_vec  


def obs_list_to_state_vector(observation):
    """ Concat indiv obs into a single np array 
    observation -> list of np arrays, contains local obs (1-D) of all agents:
    list([ array([x,y,z,...]), array([x',y',z'',...]),...])->array([x,y,z,...,x',y',z',...])
    This func only works for a single transition (no batch)                                       
    """
    state = np.array([])  # 1D
    for obs in observation:
        state = np.concatenate([state, obs], axis=0)   # concat along dim 0 (default)
    return state 

def obs_list_to_state_vector_2D(observation):
    """ Concat indiv obs into a single np array 
    This func only works for a batched transitions (2D array)                                      
    """
    state = observation[0]  #2D
    for i in range(1, len(observation)):
        state = np.concatenate([state, observation[i]], axis=1)   
    return state 


def index_2_obs(x):  
    """  convert obs index to obs list  """
    x = np.array(x).item()    # x can be an one-entry array/list
    if x == 0:
        y = [np.array([-1.0]), np.array([-1.0])]
    if x == 1:
        y = [np.array([-1.0]), np.array([1.0])]
    if x == 2:  
        y = [np.array([1.0]), np.array([-1.0])]
    if x == 3:  
        y = [np.array([1.0]), np.array([1.0])]

    return [y[0].astype('float32'), y[1].astype('float32')]    # return a list of np arrays


def move_avg_decay(in_vec, window_size=100, weight=0.9):
    """Moving average with exponential decay
        INPUT:
            in_vec, 1D
        OUTPUT:
            out_vec[i] = in_vec[i] + weight*in_vec[i+1] + weight^2*in_vec[i+2] +....
    """
    length = len(in_vec)
    in_vec = np.array(in_vec)
    out_vec = np.zeros(length)
    for i in range(length):
        if i <= length - window_size:
            out_vec[i] = sum([weight**(k-i) * in_vec[k] for k in range(i, window_size+i)])
        else:
            out_vec[i] = sum([weight**(k-i) * in_vec[k] for k in range(i, length)])
    return out_vec


def decompose_index(index_4):
    """ convert 0,1,2,3 into binary arrays
    output type: list """
    index_4 = int(index_4)
    if index_4 == 0:
        out = [0, 0]
    if index_4 == 1:
        out = [0, 1]
    if index_4 == 2:
        out = [1, 0]
    if index_4 == 3:
        out = [1, 1]
    return out    


def action_idx_2_val(idx, MIN=0.0, MAX=1.0, n_levels=10):
    """ divide interval [min_val, max_val] into n_levels discrete levels, 
        find the value w/ index = x    """
    temp = [(MIN + i*(MAX - MIN)/(n_levels - 1)) for i in range(n_levels)] 
    return temp[int(idx)]


def state_idx_2_vec(x):
    """ convert state index 0,1,2,3 into state vectors"""
    x = int(x)
    if x == 0:
        out = [-1, -1]
    if x == 1:
        out = [-1, 1] 
    if x == 2:
        out = [1, -1]
    if x == 3:
        out = [1, 1]
    return out

def find_state_percentile(x, A, I_quan, I_q):
    y = 0
    temp_percentage = np.percentile(A, I_quan)
    if x >= temp_percentage[-1]:
        y = I_q-1
    else:
        for i in range(I_q-1):
            if temp_percentage[i] <= x < temp_percentage[i + 1]:
                y = i + 1
    return int(y)
    
    
