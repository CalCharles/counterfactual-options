# generate traces
import numpy as np
import os, cv2, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from file_management import save_to_pickle, load_from_pickle
from Networks.network import ConstantNorm, pytorch_model
from tianshou.data import Collector, Batch, ReplayBuffer


def get_proximal_indexes(full_model, mask, rollouts, max_proximity):
    fullstate = rollouts.get_values("state")
    mask = pytorch_model.wrap(mask, cuda=rollouts.iscuda)
    print(mask, full_model.zeta(fullstate))
    control_feature = pytorch_model.unwrap(full_model.zeta(fullstate) * mask)
    target_feature = pytorch_model.unwrap(full_model.delta(fullstate) * mask)
    dist = np.linalg.norm(control_feature - target_feature, axis=1)
    proximal = np.zeros(dist.shape)
    proximal[dist <= max_proximity] = 1
    non_proximal = np.ones(dist.shape)
    non_proximal[dist <= max_proximity] = 0
    print(proximal[:100], non_proximal[:100])
    return proximal, non_proximal


def bin_trace(full_model, names, tr):
    if full_model.multi_instanced: # returns a vector of length instances which is 1 where there is an interaction with the desired object
        return np.array([float(len([n for n in trace if n in names]) > 0) for trace in tr])
    else:
        trace = [t for it in tr for t in it] # flattens out instances
        if len([name for name in trace if name in names]) > 0:
            return 1
        return 0

def set_traces(full_model, flat_state, names, target_name):
    # sets trace for one state, if the target has an interaction with at least one name in names
    factored_state = full_model.environment_model.unflatten_state(pytorch_model.unwrap(flat_state), vec=False, instanced=False)
    full_model.environment_model.set_interaction_traces(factored_state)
    # cv2.imshow('frame', full_model.environment_model.environment.render())
    # cv2.waitKey(100)
    tr = full_model.environment_model.get_interaction_trace(target_name[0])
    return bin_trace(full_model, names, tr)

def check_current_trace(full_model, source, target, iscuda):
    tr = full_model.environment_model.get_interaction_trace(target)
    return pytorch_model.wrap(full_model.bin_trace([source], tr), cuda=iscuda)

def generate_interaction_trace(full_model, rollouts, names, target_name):
    ''' 
    a trace basically determines if an interaction occurred, based on the "traces"
    in the environment model that record true object interactions
    '''
    print("env name", full_model.env_name, full_model.environment_model, full_model.environment_model.environment.name)
    if full_model.env_name == "Breakout":
        traces = []
        for state in rollouts.get_values("state"):
            traces.append(set_traces(full_model, state, names, target_name))
    elif full_model.env_name == "RoboStick":
        traces = list()
        for val in rollouts.get_values("info"):
            traces.append(val)

            # locality = np.array([i - self.interaction_distance for i in range(int(self.interaction_distance * 2 + 1))])
            # locality[self.interaction_distance+1:] = 0
            # traces = np.convolve(traces, locality, 'same')
            # traces = self.interaction_distance - traces


    return pytorch_model.wrap(traces, cuda=full_model.iscuda)

def adjust_interaction_trace(interaction_distance, traces, iscuda):
    if interaction_distance > 0:
        traces = pytorch_model.unwrap(traces)
        locality = np.array([np.exp(-abs(i - interaction_distance) / 4) for i in range(int(interaction_distance * 2 + 1))])
        locality[interaction_distance+1:] = 0 # set the midpoint to be 1
        traces = np.convolve(traces, locality, 'same')
        print(traces[:100])
        print(traces[100:200])
    return pytorch_model.wrap(traces, cuda=iscuda)
