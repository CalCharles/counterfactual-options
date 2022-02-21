import torch
import numpy as np
from DDPG import DDPG
from utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, k_level, H, state_dim, action_dim, render, threshold, 
                 action_bounds, action_offset, state_bounds, state_offset, lr):

        # adding lowest level
        self.HAC = [HAC_policy(state_dim, action_dim, action_bounds, action_offset, lr, H)]
        self.replay_buffer = [ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])]
        
        # adding remaining levels
        for _ in range(k_level-1):
            self.HAC.append(HAC_policy(state_dim, state_dim, state_bounds, state_offset, lr, H))
            self.replay_buffer.append(ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1]))
        
        # set some parameters
        self.low_level_no_inst = low_level_no_inst
        self.k_level = k_level
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render
        
        # logging parameters
        self.goals = [None]*self.k_level
        self.reward = 0
        self.timestep = 0
        
    def set_parameters(self, lamda, gamma, action_clip_low, action_clip_high, 
                       state_clip_low, state_clip_high, exploration_action_noise, exploration_state_noise):
        
        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise

    def get_obs(self, level, full_state, param, environment_model):

    def get_target(self, level, full_state, environment_model):
    
    
    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i]-goal[i]) > threshold[i]:
                return False
        return True
    
    def update(self, n_iter, batch_size):
        for i in range(self.k_level):
            self.HAC[i].update(self.replay_buffer[i], n_iter, batch_size)
    
    
    def save(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].save(directory, name+'_level_{}'.format(i))
    
    
    def load(self, directory, name):
        for i in range(self.k_level):
            self.HAC[i].load(directory, name+'_level_{}'.format(i))
