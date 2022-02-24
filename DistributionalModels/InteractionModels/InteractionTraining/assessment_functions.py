# assessment functions
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
from DistributionalModels.InteractionModels.InteractionTraining.compute_errors import get_target_magnitude, get_prediction_error, get_error, get_interaction_vals, get_binaries
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import get_weights, get_targets

from Rollouts.rollouts import ObjDict, merge_rollouts


def compute_interaction_stats(full_model, rollouts, trace=None, passive_error_cutoff=2, max_distance_epsilon=0, position_mask=None):
    ints = get_interaction_vals(full_model, rollouts, multi=True)
    bins, fe, pe = get_binaries(full_model, rollouts)
    # if full_model.multi_instanced: bins = torch.max(bins, )
    if trace is None:
        trace = generate_interaction_trace(full_model, rollouts, [full_model.control_feature], [full_model.target_name])
    traceidx, intsidx = trace, ints
    if full_model.multi_instanced: 
        traceidx = torch.max(trace, dim=1)[0].squeeze()
        intsidx = np.max(ints, axis=1)[0].squeeze()
    trace = pytorch_model.unwrap(trace)
    passive_error = get_prediction_error(full_model, rollouts)
    proximal = None
    if max_distance_epsilon > 0:
        proximal, non_proximal = get_proximal_indexes(full_model, position_mask, rollouts, max_distance_epsilon)
        non_proximal_weights = non_proximal / np.sum(non_proximal)
    weights, use_weights, total_live, total_dead, ratio_lambda = get_weights(passive_error, ratio_lambda=1, passive_error_cutoff=passive_error_cutoff, use_proximity=proximal)     
    print(ints.shape, bins.shape, trace.shape, fe.shape, pe.shape)
    pints, ptrace = np.zeros(ints.shape), np.zeros(trace.shape)
    pints[ints > .7] = 1
    ptrace[trace > 0] = 1
    # print_weights = (weights + pints.squeeze() + ptrace).squeeze()
    # print_weights[print_weights > 1] = 1

    print(ints.shape, bins.shape, np.expand_dims(trace, 1).shape, fe.shape, pe.shape)
    if not full_model.multi_instanced: trace = np.expand_dims(trace, 1)
    comb = np.concatenate([ints, bins, trace, fe, pe], axis=1)
    
    bin_error = bins.squeeze()-trace.squeeze()
    bin_false_positives = np.sum(bin_error[bin_error > 0])
    bin_false_negatives = np.sum(np.abs(bin_error[bin_error < 0]))

    int_bin = ints.copy()
    int_bin[int_bin >= .5] = 1
    int_bin[int_bin < .5] = 0
    int_error = int_bin.squeeze() - trace.squeeze()
    # int_positives = np.sum(int_error.unsqueeze(1)[int_bin.squeeze() > 0])
    int_false_positives = np.sum(int_error[int_error > 0])
    int_false_negatives = np.sum(np.abs(int_error[int_error < 0]))
    
    dstate = pytorch_model.unwrap(full_model.gamma(rollouts.get_values("state")))
    dnstate = pytorch_model.unwrap(full_model.delta(get_targets(full_model.predict_dynamics, rollouts)))
    print("int false positives", np.concatenate((dstate[int_error > 0], dnstate[int_error > 0], ints[int_error > 0], trace[int_error > 0], fe[int_error>0], pe[int_error>0]), axis=-1)[:100])
    print("int false negatives", np.concatenate((dstate[int_error < 0], dnstate[int_error < 0], ints[int_error < 0], trace[int_error < 0], fe[int_error<0], pe[int_error<0]), axis=-1)[:100])

    comb_error = bins.squeeze() + int_bin.squeeze()
    comb_error[comb_error > 1] = 1
    comb_error = comb_error - trace.squeeze()
    comb_false_positives = np.sum(comb_error[comb_error > 0])
    comb_false_negatives = np.sum(np.abs(comb_error[comb_error < 0]))

    print("bin fp, fn", bin_false_positives, bin_false_negatives)
    print("int fp, fn", int_false_positives, int_false_negatives)
    print("com fp, fn", comb_false_positives, comb_false_negatives)
    print("total, tp", trace.shape[0], np.sum(trace))
    del bins
    del fe
    del pe
    del pints
    del ptrace
    del comb
    del bin_error
    del bin_false_positives
    del bin_false_negatives
    del comb_error

def assess_error(full_model, test_rollout, passive_error_cutoff=2, trace=None, max_distance_epsilon=0, position_mask=None):
    print("assessing_error", test_rollout.filled)
    if full_model.env_name == "Breakout": 
        compute_interaction_stats(full_model, test_rollout, passive_error_cutoff=passive_error_cutoff, trace=trace, max_distance_epsilon=max_distance_epsilon, position_mask=position_mask)
    rv = full_model.output_normalization_function.reverse
    states = test_rollout.get_values("state")
    interaction, forward, passive = list(), list(), list()
    for i in range(int(np.ceil(test_rollout.filled / 100))):
        inter, f, p = full_model.hypothesize(states[i*100:(i+1)*100])
        interaction.append(pytorch_model.unwrap(inter)), forward.append(pytorch_model.unwrap(f)), passive.append(pytorch_model.unwrap(p))
    interaction, forward, passive = np.concatenate(interaction, axis=0), np.concatenate(forward, axis=0), np.concatenate(passive, axis=0)
    targets = get_targets(full_model.predict_dynamics, test_rollout)
    dtarget = full_model.split_instances(full_model.delta(targets)) if full_model.multi_instanced else full_model.delta(targets)
    axis = 2 if full_model.multi_instanced else 1
    print(forward.shape, dtarget.shape, interaction.shape)
    inter_bin = interaction.copy()
    inter_bin[interaction >= full_model.interaction_prediction] = 1
    inter_bin[interaction < full_model.interaction_prediction] = 0
    sfe = np.linalg.norm(forward - pytorch_model.unwrap(dtarget), ord =1, axis=axis) * interaction.squeeze() # per state forward error
    spe = np.linalg.norm(passive - pytorch_model.unwrap(dtarget), ord =1, axis=axis) * interaction.squeeze() # per state passive error
    # print(full_model.output_normalization_function.mean, full_model.output_normalization_function.std)
    cat_ax = 1
    if full_model.multi_instanced:
        interaction = np.expand_dims(interaction, axis=2)
        cat_ax = 2
    print(forward[:100].shape, (forward - pytorch_model.unwrap(dtarget))[:100].shape, (passive - pytorch_model.unwrap(dtarget))[:100].shape, pytorch_model.unwrap(dtarget)[:100].shape, pytorch_model.unwrap(interaction)[:100].shape)
    print("forward, passive, interaction", np.concatenate([forward[:100], (forward - pytorch_model.unwrap(dtarget))[:100], (passive - pytorch_model.unwrap(dtarget))[:100], pytorch_model.unwrap(dtarget)[:100], pytorch_model.unwrap(interaction)[:100]], axis=cat_ax))
    sfeat = sfe[interaction.squeeze() > .5]
    speat = spe[interaction.squeeze() > .5]
    dtargetat = dtarget[interaction.squeeze() > .5]
    interat = interaction[interaction.squeeze() > .5]
    print([np.expand_dims(sfeat[:100], 1).shape, np.expand_dims(speat[:100], 1).shape, pytorch_model.unwrap(dtargetat)[:100].shape, interat[:100].shape])
    print("forward, passive, inter-normalized", np.concatenate([np.expand_dims(sfeat[:100], 1), np.expand_dims(speat[:100], 1), pytorch_model.unwrap(dtargetat)[:100], interat[:100]], axis=1))
    print("inputs", full_model.gamma(test_rollout.get_values("state"))[:100])
    print("targets", full_model.delta(targets)[:100])
    forward_error = np.sum(np.abs(sfe)) / np.sum(interaction)
    passive_error = np.sum(np.abs(spe)) / np.sum(interaction)
    print("comparison", forward_error, passive_error)
    return forward_error, passive_error

