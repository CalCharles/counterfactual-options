# sampling methods
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
from DistributionalModels.InteractionModels.InteractionTraining.traces import set_traces
from DistributionalModels.InteractionModels.state_management import StateSet


def collect_samples(full_model, rollouts, use_trace=False):
    # collecs the possible states to sample, based on where the interaction model gives > interaction_prediction probability
    full_model.sample_able = StateSet()
    for state,next_state in zip(rollouts.get_values("state"), rollouts.get_values("next_state")):
        if use_trace:
            inter = set_traces(state, [full_model.control_feature], [full_model.target_name])
        else:
            inter = full_model.interaction_model(full_model.gamma(state))
        if inter > full_model.interaction_prediction:
            inputs, targets = [full_model.gamma(state)], [full_model.delta(next_state)]
            if full_model.multi_instanced:
                inter_bin = full_model.interaction_model.instance_labels(full_model.gamma(state))
                inter_bin[inter_bin<.2] = 0
                idxes = inter_bin.nonzero()
                mvtg = full_model.split_instances(full_model.delta(next_state))
                inputs, targets = list(), list()
                # print(inter_bin.shape)
                for idx in idxes:
                    # print(inter_bin[0, idx[1]])
                    # print(idx, mvtg.shape, inter_bin.shape)
                    targets.append(mvtg[idx[1]])
            print("sample", inter, inputs, targets)
            for tar in targets:
                sample = pytorch_model.unwrap(tar) * pytorch_model.unwrap(full_model.selection_binary)
                full_model.sample_able.add(sample)
    # if full_model.iscuda:
    #     full_model.sample_able.cuda()
    print(full_model.sample_able.vals)

def sample(self, states):
    if self.sample_continuous: # TODO: states should be a full environment state, so need to apply delta to get the appropriate parts
        weights = np.random.random((len(self.cfselectors,))) # random weight vector
        lower_cfs = np.array([i for i in [cfs.feature_range[0] for cfs in self.cfselectors]])
        len_cfs = np.array([j-i for i,j in [tuple(cfs.feature_range) for cfs in self.cfselectors]])
        edited_features = lower_cfs + len_cfs * weights
        new_states = copy.deepcopy(states)
        for f, cfs in zip(edited_features, self.cfselectors):
            cfs.assign_feature(new_states, f)
        if len(new_states.shape) > 1: # if a stack, duplicate mask for all
            return self.delta(new_states), pytorch_model.wrap(torch.stack([self.selection_binary.clone() for _ in range(new_states.size(0))], dim=0), cuda=self.iscuda)
        return self.delta(new_states), self.selection_binary.clone()
    else: # sample discrete with weights, only handles single item sampling
        value = np.random.choice(self.sample_able.vals)
        return value.clone(), self.selection_binary.clone()
