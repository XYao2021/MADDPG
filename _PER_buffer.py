"""Prioritized Experience Replay (PER) buffer"""
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""/////////// Replay Buffer /////////////"""

class PER_MultiAgentReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        """ critic_dims: (flattened) state vector, size =(obs_len*n_agents) """
        self.mem_size = max_size  # buffer size
        self.mem_cntr = 0  # buffer counter
        self.n_agents = n_agents
        self.actor_dims = actor_dims  # list of each agent's obs dim, size =(n_agents,)
        self.batch_size = batch_size
        self.n_actions = n_actions
        # number of action dims for each agent (each action dim is continuous)
        self.state_memory = np.zeros((self.mem_size, critic_dims))  # flattened state vector
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)
        # terminal states have zero Q-values
        self.init_actor_memory()  # init actor memory, list of arrays

    def init_actor_memory(self):
        """ init buffer for each agent """
        self.actor_state_memory = []  # list of (obs_len, ) arrays
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))  # obs input to actor i
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(
                np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size
        """ raw_obs: list of local obs vectors, size =(n_agents, obs_len)
            state: flattened state vector
            action: vec of agent actions, size =(n_agent,) """
        for agent_idx in range(self.n_agents):  # store trans for actors
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]
        self.state_memory[index] = state  # store trans for critic
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward  # reward: 1D array
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)  # returns 1D array
        """for critic use"""
        states = self.state_memory[batch]  # np array, size =(batch_size,critic_dims)
        rewards = self.reward_memory[batch]  # size = (batch_size, n_agents)
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]  # size = (batch_size, n_agents)
        """for actor use"""
        actor_states = []  # list of arrays (size=(batch_size,self.actor_dims[i])) for agent i
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return actor_states, states, actions, rewards, actor_new_states, states_, terminal

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False  # do not learn if samples less than a batch