# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:33:29 2019
adapted from: https://github.com/orrivlin/Navigation-HER.git
@author: Or
"""

import torch
import numpy as np
import gym
# from matplotlib.pyplot import imshow
# import matplotlib as plt
from copy import deepcopy as dc
from Environments.environment_specification import RawEnvironment




class Nav2D(RawEnvironment):
    def __init__(self,frameskip=1,N=20,Nobs=0,Dobs=2,Rmin=10):
        self.seed_counter = -1

        self.N = N
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.Rmin = Rmin
        self.state_dim = [N,N,3]
        self.num_actions = 4
        self.action_space = gym.spaces.Discrete(4)
        self.action_shape = self.action_space.shape
        self.scale = 10.0
        self.itr = 0
        self.save_path = ""
        self.recycle = -1
        self.frameskip = 1
        self.done = False
        self.action = np.array(0)
        # if len(self.action.shape) == 0:
        #     self.action = np.array([self.action])
        full_state = self.reset()
        self.frame = full_state["raw_state"]
        self.reward = -1
        self.reshape = [N,N,3]
        self.trajectory_len = 50
        self.discrete_actions = True

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(N, N, 3))
        self.observation_shape = self.state_dim
        self.done = 0
        self.reward = 0

        self.extracted_state = self.extracted_state_dict()

    def param_process(self, obs, param):
        for i in range(obs.shape[0]):
            obs[i,:,:,2] = param[i]
        return obs

    def preprocess(self, obs):
        obs = obs.transpose((0,3,1,2))
        return obs


    def get_dims(self):
        return self.state_dim, self.num_actions
        
    def reset(self):
        self.frame = np.zeros((self.N,self.N,3))
        for i in range(self.Nobs):
            center = np.random.randint(0,self.N,(1,2))
            minX = np.maximum(center[0,0] - self.Dobs,1)
            minY = np.maximum(center[0,1] - self.Dobs,1)
            maxX = np.minimum(center[0,0] + self.Dobs,self.N-1)
            maxY = np.minimum(center[0,1] + self.Dobs,self.N-1)
            self.frame[minX:maxX,minY:maxY,0] = 1.0
            
        free_idx = np.argwhere(self.frame[:,:,0] == 0.0)
        start = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
        while (True):
            finish = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
            if ((start[0] != finish[0]) and (start[1] != finish[1]) and (np.linalg.norm(start - finish) >= self.Rmin)):
                break
        # start = np.array([1,1])
        # finish = np.array([15,18])
        self.pos = start
        self.target = finish
        self.reward = -1
        self.done = False
        self.frame[start[0],start[1],1] = self.scale*1.0
        self.frame[finish[0],finish[1],2] = self.scale*1.0
        done = False
        return {"raw_state": self.frame, "factored_state": self.extracted_state_dict()}

    def check_reward(self, input_state, state, param):
        frame = input_state
        rews = list()
        for p, frame in zip(param, input_state):
            target = np.argwhere(p == self.scale*1.0)[0]
            pos = np.argwhere(frame[:,:,1] == self.scale**1.0)[0]
            rews.append(int(np.linalg.norm(target - pos) < .0001) - 1)
        return np.array(rews) 

    def check_done(self, input_state, state, param):
        frame = input_state
        dones = list()
        for p, frame in zip(param, input_state):
            target = np.argwhere(p == self.scale*1.0)[0]
            pos = np.argwhere(frame[:,:,1] == self.scale**1.0)[0]
            dones.append(np.linalg.norm(target - pos) < .0001)
        return np.array(dones)

    def step(self,action):
        max_norm = self.N
        action = int(action)

        new_frame = dc(self.frame)
        self.done = False
        self.reward = -1.0
        act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        pos = np.argwhere(self.frame[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(self.frame[:,:,2] == self.scale*1.0)[0]
        new_pos = pos + act[action]
        self.action = action
        self.target = target
        
        dist1 = np.linalg.norm(pos - target)
        self.dist2 = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)*(max_norm - dist2)
        #reward = -dist2
        self.reward = -1
        self.itr += 1
        if self.itr % self.trajectory_len == 0 and self.itr != 0:
            # print("failure")
            frame = dc(self.frame)
            self.reset()
            self.done = True
            return {'raw_state': frame, 'factored_state': self.extracted_state_dict()}, self.reward, True, {"TimeLimit.truncated": True}
        if (np.any(new_pos < 0.0) or np.any(new_pos > (self.N - 1)) or (self.frame[new_pos[0],new_pos[1],0] == 1.0)):
            #dist = np.linalg.norm(pos - target)
            #reward = (dist1 - dist2)
            extracted_state = self.extracted_state_dict()
            return {'raw_state': self.frame, 'factored_state': extracted_state}, self.reward, self.done, {"TimeLimit.truncated": False}
        self.pos = new_pos
        new_frame[pos[0],pos[1],1] = 0.0
        new_frame[new_pos[0],new_pos[1],1] = self.scale*1.0
        done = False
        if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
            self.reward = 0.0
            done = True
            # print("success")
            # print("reached goal", self.itr)
        #dist = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)
        self.done = done
        extracted_state = self.extracted_state_dict()
        self.frame = new_frame
        if len(self.save_path) != 0:
            if self.itr == 0:
                object_dumps = open(os.path.join(self.save_path, "object_dumps.txt"), 'w') # create file if it does not exist
                object_dumps.close()
            self.write_objects(extracted_state, self.frame)
        if done:
            # print("done", self.itr, self.done)
            frame = dc(self.frame)
            self.reset()
            self.reward = 0.0
            return {'raw_state': frame, 'factored_state': extracted_state}, self.reward, True, {"TimeLimit.truncated": False}
        return {'raw_state': self.frame, 'factored_state': extracted_state}, self.reward, self.done, {"TimeLimit.truncated": False}

    def get_state(self):
        return {'raw_state': self.frame, 'factored_state': self.extracted_state_dict()}

    def toString(self, extracted_state):
        names = ["Action", "Pos", "Target", "Reward", "Done"]
        es = ""
        for name in names:
            es += name + ":" + " ".join(map(str, extracted_state[name])) + "\t"
        return es

    def extracted_state_dict(self):
        return {"Action": np.array([0,0,self.action]), 
            "Pos": np.array([self.pos[0], self.pos[1], 1]), 
            "Target": np.array([self.target[0], self.target[1], 1]),
            "Frame": self.frame.flatten(),
            "Reward": np.array([self.reward]),
            "Done": np.array([float(self.done)])}
    
    def get_tensor(self,frame):
        S = torch.Tensor(frame).transpose(2,1).transpose(1,0).unsqueeze(0)
        return S
    
    def render_frame(self):
        #imshow(frame)
        # plot = imshow(self.frame)
        # return plot
        return self.frame