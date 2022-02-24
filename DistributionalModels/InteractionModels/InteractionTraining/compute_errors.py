# error manager
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
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import get_targets


def get_error(output_norm_fn, mean, target, normalized = False):
    rv = output_norm_fn.reverse # self.output_normalization_function.reverse
    nf = output_norm_fn # self.output_normalization_function
    # print((rv(mean) - target).abs().sum(dim=1).shape)
    if normalized:
        # print(mean, nf(target))
        return (mean - nf(target)).abs().sum(dim=1).unsqueeze(1)
    else:
        return (rv(mean) - target).abs().sum(dim=1).unsqueeze(1)

def get_prediction_error(full_model, rollouts, active=False):
    pred_error = []
    nf = full_model.output_normalization_function
    done_flags = 1-rollouts.get_values("done")
    if active:
        dstate = full_model.gamma(rollouts.get_values("state"))
        model = full_model.forward_model
    else:
        dstate = full_model.delta(rollouts.get_values("state"))
        model = full_model.passive_model
    dnstate = full_model.delta(get_targets(full_model.predict_dynamics, rollouts))
    for i in range(int(np.ceil(rollouts.filled / 500))): # run 10k at a time
        dstate_part = dstate[i*500:(i+1)*500].cuda() if full_model.iscuda else dstate[i*500:(i+1)*500]
        dnstate_part = dnstate[i*500:(i+1)*500].cuda() if full_model.iscuda else dnstate[i*500:(i+1)*500]
        pred_error.append(pytorch_model.unwrap(get_error(full_model.output_normalization_function, model(dstate_part)[0], dnstate_part)) * pytorch_model.unwrap(done_flags[i*500:(i+1)*500]))
        # passive_error.append(full_model.dist(*full_model.passive_model(dstate[i*10000:(i+1)*10000])).log_prob(nf(dnstate[i*10000:(i+1)*10000])))
    return np.concatenate(pred_error, axis=0)


def get_target_magnitude(get_targets, predict_dynamics, rollouts):
    l1mag = list()
    targets = get_targets(predict_dynamics, rollouts) # self.get_targets
    for i in range(int(np.ceil(rollouts.filled / 500))): # run 10k at a time
        l1mag.append(pytorch_model.unwrap(targets[i*500:(i+1)*500].norm(p=1, dim=1)))
    return np.concatenate(l1mag, axis=0)


def get_interaction_vals(full_model, rollouts, multi=False):
    interaction = []
    for i in range(int(np.ceil(rollouts.filled / 500))):
        inputs = rollouts.get_values("state")[i*500:(i+1)*500]
        if full_model.multi_instanced and multi: ints = full_model.interaction_model.instance_labels(full_model.gamma(inputs))
        else: ints = full_model.interaction_model(full_model.gamma(inputs))
        interaction.append(pytorch_model.unwrap(ints))
    return np.concatenate(interaction, axis=0)


def get_binaries(full_model, rollouts):
    bins = []
    rv = full_model.output_normalization_function.reverse
    fe, pe = list(), list()
    for i in range(int(np.ceil(rollouts.filled / 500))):
        inputs = rollouts.get_values("state")[i*500:(i+1)*500].cuda() if full_model.iscuda else rollouts.get_values("state")[i*500:(i+1)*500]
        targets = get_targets(full_model.predict_dynamics, rollouts)[i*500:(i+1)*500].cuda() if full_model.iscuda else get_targets(full_model.predict_dynamics, rollouts)[i*500:(i+1)*500]
        prediction_params = full_model.forward_model(full_model.gamma(inputs))
        interaction_likelihood = full_model.interaction_model(full_model.gamma(inputs))
        passive_prediction_params = full_model.passive_model(full_model.delta(inputs))
        target = full_model.output_normalization_function(full_model.delta(targets))
        passive_loss = - full_model.dist(*passive_prediction_params).log_prob(target)
        forward_error = - full_model.dist(*prediction_params).log_prob(target)
        if full_model.multi_instanced: passive_loss = full_model.split_instances(passive_loss).sum(dim=2)
        else: passive_loss = passive_loss.sum(dim=1).unsqueeze(1)
        if full_model.multi_instanced: forward_error = full_model.split_instances(forward_error).sum(dim=2)
        else: forward_error = forward_error.sum(dim=1).unsqueeze(1)
        interaction_binaries, potential = full_model.compute_interaction(forward_error, passive_loss, rv(target))
        # interaction_binaries, potential = full_model.compute_interaction(prediction_params[0].clone().detach(), passive_prediction_params[0].clone().detach(), rv(target))
        bins.append(pytorch_model.unwrap(interaction_binaries))
        fe.append(pytorch_model.unwrap(forward_error))
        pe.append(pytorch_model.unwrap(passive_loss))
    return np.concatenate(bins, axis=0), np.concatenate(fe, axis=0), np.concatenate(pe, axis=0)

def assess_losses(full_model, test_rollout):
    forward_loss, passive_loss = list(), list()
    for i in range(int(np.ceil(test_rollout.filled / 200))):
        if i % 100 == 0: print("assessing at: ", i * 200)
        state, target = test_rollout.get_values("state")[i*200:(i+1)*200], get_targets(full_model.predict_dynamics, test_rollout)[i*200:(i+1)*200]
        prediction_params = full_model.forward_model(full_model.gamma(state))
        interaction_likelihood = full_model.interaction_model(full_model.gamma(state))
        passive_prediction_params = full_model.passive_model(full_model.delta(state))
        passive_loss.append(pytorch_model.unwrap(- full_model.dist(*passive_prediction_params).log_prob(full_model.delta(target)) * interaction_likelihood))
        forward_loss.append(pytorch_model.unwrap(- full_model.dist(*prediction_params).log_prob(full_model.delta(target)) * interaction_likelihood))
    pl, fl = np.concatenate(passive_loss, axis=0).mean(), np.concatenate(forward_loss, axis=0).mean()
    print("passive, forward losses", pl, fl)
    return pl, fl
