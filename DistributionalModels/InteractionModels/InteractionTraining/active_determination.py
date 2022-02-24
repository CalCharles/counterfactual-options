# determine active set
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
from DistributionalModels.InteractionModels.InteractionTraining.traces import get_proximal_indexes, generate_interaction_trace, adjust_interaction_trace
from DistributionalModels.InteractionModels.InteractionTraining.train_passive import train_passive
from DistributionalModels.InteractionModels.InteractionTraining.train_interaction import train_interaction
from DistributionalModels.InteractionModels.InteractionTraining.train_combined import train_combined
from DistributionalModels.InteractionModels.InteractionTraining.compute_errors import get_target_magnitude, get_prediction_error, get_error, get_interaction_vals
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import get_weights, get_targets

from Rollouts.rollouts import ObjDict, merge_rollouts
from EnvironmentModels.environment_model import get_selection_list, FeatureSelector, ControllableFeature, sample_multiple



def test_forward(full_model, states, next_state, interact=True):
    # gives back the difference between the prediction mean and the actual next state for different sampled feature values
    rv = full_model.output_normalization_function.reverse
    checks = list()
    print(np.ceil(len(states)/2000))
    batch_pred, inters = list(), list()
    # painfully slow when states is large, so an alternative might be to only look at where inter.sum() > 1
    for state in states:
        # print(full_model.gamma(full_model.control_feature.sample_feature(state)))
        control = full_model.controllers
        if type(full_model.controllers) == list and len(full_model.controllers) == 1:
            control = full_model.controllers[0]
        if type(control) == list: # multiple control features
            sampled_feature = sample_multiple(full_model.controllers, pytorch_model.unwrap(state))
        else:
            sampled_feature = control.sample_feature(state)
        # print(full_model.gamma(sampled_feature))
        # print(sampled_feature.shape, print(type(full_model.control_feature)))
        sampled_feature = pytorch_model.wrap(sampled_feature, cuda=full_model.iscuda)
        inter, pred_states = full_model.predict_next_state(sampled_feature)
        # if inter.sum() > .7:
        #     # print(inter.shape, pred_states.shape, sampled_feature.shape, inter > 0)
        #     print(inter[inter.squeeze() > 0.7], pred_states[inter.squeeze() > 0.7], full_model.gamma(sampled_feature[inter.squeeze() > 0.7]), full_model.control_feature.object())
        # if inter.sum() >= 1:
        #     print('int', pytorch_model.unwrap(inter))
        #     print('sam', pytorch_model.unwrap(full_model.gamma(sampled_feature)))
        #     print('pred_states', pytorch_model.unwrap(pred_states))
        batch_pred.append(pred_states.cpu().clone().detach()), inters.append(inter.cpu().clone().detach())
        del pred_states
        del inter
    batch_pred, inters = torch.stack(batch_pred, dim=0), torch.stack(inters, dim=0) # batch x samples x state, batch x samples x 1
    next_state_broadcast = pytorch_model.wrap(torch.stack([full_model.delta(next_state).clone().cpu() for _ in range(batch_pred.size(1))], dim=1)).cpu()
    # compare predictions with the actual next state to make sure there are differences
    print(int(np.ceil(len(states)/2000)), batch_pred, next_state_broadcast)
    state_check = (next_state_broadcast - batch_pred).abs()
    print("state_check", state_check[:10])
    # should be able to predict at least one of the next states accurately
    match = state_check.min(dim=1)[0]
    match_keep = match.clone()
    print(match[:10], full_model.interaction_prediction)
    match_keep[match <= full_model.interaction_prediction] = 1
    match_keep[match > full_model.interaction_prediction] = 0
    if interact: # if the interaction value is less, assume there is no difference because the model is flawed
        inters[inters > full_model.interaction_prediction] = 1
        inters[inters <= full_model.interaction_prediction] = 0
        checks.append((state_check * match_keep.unsqueeze(1)) * inters) # batch size, num samples, state size
    else:
        checks.append(state_check * match_keep.unsqueeze(1))
    return torch.cat(checks, dim=0)


def determine_active_set(full_model, rollouts, use_hardcoded_rng=None, feature_step=1):
    states = rollouts.get_values('state')
    next_states = rollouts.get_values('state')
    targets = get_targets(full_model.predict_dynamics, rollouts)
    # create a N x num samples x state size of the nth sample tested for difference on the num samples of assignments of the controllable feature
    # then take the largest difference along the samples
    sample_diffs = torch.max(test_forward(full_model, states, next_states), dim=1)[0]
    # take the largest difference at any given state
    test_diff = torch.max(sample_diffs, dim=0)[0]
    v = torch.zeros(test_diff.shape)
    # if the largest difference is larger than the active_epsilon, assign it
    print("test_diff", test_diff)
    v[test_diff > full_model.active_epsilon] = 1
    # collect by instance and determine
    v = full_model.split_instances(v)
    print(v.shape)
    v = torch.max(v, dim=0)[0]

    print("act set", v, v.sum())
    if v.sum() == 0:
        return None, None

    # create a feature selector to match that
    full_model.selection_binary = pytorch_model.wrap(v, cuda=full_model.iscuda)
    full_model.feature_selector, full_model.reverse_feature_selector = full_model.environment_model.get_factored_subset(full_model.delta, v)

    # create a controllable feature selector for each controllable feature
    full_model.cfselectors = list()
    for ff in full_model.feature_selector.flat_features:
        factored = full_model.environment_model.flat_to_factored(ff)
        print(factored)
        single_selector = FeatureSelector([ff], {factored[0]: factored[1]}, {factored[0]: np.array([factored[1], ff])}, [factored[0]])
        rng = determine_range(rollouts, single_selector, use_hardcoded_rng = use_hardcoded_rng)
        print(rng)
        full_model.cfselectors.append(ControllableFeature(single_selector, rng, feature_step, full_model))
    full_model.selection_list = get_selection_list(full_model.cfselectors)
    full_model.control_min = [cfs.feature_range[0] for cfs in full_model.cfselectors]
    full_model.control_max = [cfs.feature_range[1] for cfs in full_model.cfselectors]
    return full_model.feature_selector, full_model.cfselectors

def determine_range(rollouts, active_delta, use_hardcoded_rng):
    # Even if we are predicting the dynamics, we determine the active range with the states
    # TODO: However, using the dynamics to predict possible state range ??
    # if self.predict_dynamics:
    #     state_diffs = rollouts.get_values('state_diff')
    #     return active_delta(state_diffs).min(dim=0)[0], active_delta(state_diffs).max(dim=0)[0]
    # else:
    if use_hardcoded_rng is not None:
        return float(pytorch_model.unwrap(active_delta(use_hardcoded_rng[0]).squeeze())), float(pytorch_model.unwrap(active_delta(use_hardcoded_rng[0]).squeeze()))
    else:
        states = rollouts.get_values('state')
        return float(pytorch_model.unwrap(active_delta(states).min(dim=0)[0].squeeze())), float(pytorch_model.unwrap(active_delta(states).max(dim=0)[0].squeeze()))
