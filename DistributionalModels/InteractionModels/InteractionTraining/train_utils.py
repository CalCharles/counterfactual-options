# general trainer
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


def get_weights(err=None, ratio_lambda=2, passive_error_cutoff=2, 
	passive_error_upper=10, weights=None, temporal_local=0, use_proximity=None):
    # if err is not none, then an error/negative likelihood is being used to generate weights
    # ratio_lambda determines how much weighting to put on the weighted values over lambda (1) and the existing ratio
    # passive error cutoff is the lower bound
    # use proximity is the binary values for states where the objects are proximal

    # determine error based binary weights
    if err is not None:
        # print(err[:100])
        # print(err[100:200])
        # print(err[200:300])

        weights = pytorch_model.unwrap(err).copy().squeeze()
        weights[weights<=passive_error_cutoff] = 0
        weights[weights>passive_error_upper] = 0
        weights[weights>passive_error_cutoff] = 1
    weights = pytorch_model.unwrap(weights)

    # convolve weights based on temporal locality to an interaction TODO: not really used
    if temporal_local:
        locality = np.array([.4 for i in range(int(temporal_local))])
        locality[int(temporal_local // 2)] = 1 # set the midpoint to be 1
        weights = np.convolve(weights, locality, 'same')

    # using the proximity assumption with weights
    if use_proximity is not None:
        print(use_proximity[:100])
        weights = (weights.astype(int) * use_proximity.astype(int)).astype(np.float128)

    # generate a ratio based on the number of live versus dead
    total_live = np.sum(weights)
    total_dead = np.sum((weights + 1)) - np.sum(weights) * 2
    live_factor = np.float128(np.round(total_dead / total_live * ratio_lambda))
    use_weights = (weights * live_factor) + 1
    print(type(live_factor), use_weights.dtype)
    print(use_weights[:100])
    print(use_weights[100:200])
    print(use_weights[200:300])
    use_weights = (use_weights / np.sum(use_weights)).astype(np.float64)
    # clean up underflow/overflow errors
    print(np.sum(use_weights), np.sum(use_weights) < 0, np.sum(use_weights) > 0, 1-np.sum(use_weights))
    if np.sum(use_weights) < 1:
        use_weights[0] += 1-np.sum(use_weights)
    elif np.sum(use_weights) > 1:
        use_weights[0] += 1-np.sum(use_weights)
    # if np.sum(use_weights) > 1:
    # if np.sum(use_weights) > 1:
    #     i=0
    #     while np.sum(use_weights) != 1:
    #         print(i, use_weights[i], np.sum(use_weights), 1-np.sum(use_weights))
    #         if use_weights[i] > 1-np.sum(use_weights):
    #             use_weights[i] -= 1-np.sum(use_weights)
    #         else:
    #             use_weights[i] = 0

    #         i += 1
    print(np.sum(use_weights), 1-np.sum(use_weights))
    print(use_weights[:100])
    # use_weights = pytorch_model.unwrap(use_weights)
    # print(weights + 1, temporal_local)
    # error
    print("live, dead, factor", total_live, total_dead, live_factor)
    return weights, use_weights, total_live, total_dead, ratio_lambda

def get_targets(predict_dynamics, rollouts):
    # the target is whether we predict the state diff or the next state
    if predict_dynamics:
        targets = rollouts.get_values('state_diff')
    else:
        targets = rollouts.get_values('next_state')
    return targets

def run_optimizer(optimizer, model, loss):
    optimizer.zero_grad()
    (loss.mean()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
