# train passive model
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
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import run_optimizer, get_targets

def _passive_logging(full_model, train_args, i, batchvals, target, passive_prediction_params, passive_loss, prediction_params, active_loss):
        if full_model.multi_instanced:
            print("Iters: ", i, ", pl: ", passive_loss.mean().detach().cpu().numpy())
            for j in range(2):
                print(
                    # full_model.network_args.normalization_function.reverse(passive_prediction_params[0][0]),
                    # full_model.network_args.normalization_function.reverse(passive_prediction_params[1][0]), 
                    "input", pytorch_model.unwrap(full_model.gamma(batchvals.values.state)[j]),
                    "\npinput", pytorch_model.unwrap(full_model.delta(batchvals.values.state[j])),
                    "ninput", pytorch_model.unwrap(full_model.inf(full_model.gamma(batchvals.values.state)[j])),
                    "\npninput", pytorch_model.unwrap(full_model.dnf(full_model.delta(batchvals.values.state[j]))),
                    # "\naoutput", pytorch_model.unwrap(full_model.rv(prediction_params[0])[j]),
                    # "\navariance", full_model.rv(prediction_params[1]),
                    "\npoutput", pytorch_model.unwrap(full_model.rv(passive_prediction_params[0])[j]),
                    "\npn_output", pytorch_model.unwrap(passive_prediction_params[0][j]),
                    "\npvariance", pytorch_model.unwrap(passive_prediction_params[1][j]),
                    # full_model.delta(batchvals.values.next_state[0]), 
                    # full_model.gamma(batchvals.values.state[0]),
                    "\ntarget: ", pytorch_model.unwrap(full_model.rv(target)[j]),
                    # "\nal: ", active_loss,
                    # "\npl: ", passive_loss
                    )
        else:
            for j in range(train_args.batch_size):
                if target[j].abs().sum() > 0:
                    print(
                        # full_model.network_args.normalization_function.reverse(passive_prediction_params[0][0]),
                        # full_model.network_args.normalization_function.reverse(passive_prediction_params[1][0]), 
                        "input", pytorch_model.unwrap(full_model.gamma(batchvals.values.state)[j]),
                        "\npinput", pytorch_model.unwrap(full_model.delta(batchvals.values.state[j])),
                        "ninput", pytorch_model.unwrap(full_model.inf(full_model.gamma(batchvals.values.state)[j])),
                        "\npninput", pytorch_model.unwrap(full_model.dnf(full_model.delta(batchvals.values.state[j]))),
                        # "\naoutput", pytorch_model.unwrap(full_model.rv(prediction_params[0])[j]),
                        # "\navariance", full_model.rv(prediction_params[1]),
                        "\npoutput", pytorch_model.unwrap(full_model.rv(passive_prediction_params[0])[j]),
                        "\npn_output", pytorch_model.unwrap(passive_prediction_params[0][j]),
                        "\npvariance", pytorch_model.unwrap(passive_prediction_params[1][j]),
                        # full_model.delta(batchvals.values.next_state[0]), 
                        # full_model.gamma(batchvals.values.state[0]),
                        "\ntarget: ", pytorch_model.unwrap(full_model.rv(target)[j]),
                        # "\nal: ", active_loss,
                        # "\npl: ", passive_loss
                        )
                    if not train_args.no_pretrain_active:
                        print("aoutput", pytorch_model.unwrap(full_model.rv(prediction_params[0])[j]),
                            "\nal: ", pytorch_model.unwrap(active_loss[j]))
        active_str = ""
        if not train_args.no_pretrain_active:
            active_str = ", al: " +  str(active_loss.mean().detach().cpu().numpy())
        print("Iters", i, ", pl: ", passive_loss.mean().detach().cpu().numpy(), active_str,
            )


def train_passive(full_model, rollouts, train_args, batchvals, active_optimizer, passive_optimizer, idxes_sets=None, keep_outputs=False, weights=None):
    outputs = list()
    for i in range(train_args.pretrain_iters):
        # get input-output values
        if idxes_sets is not None: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, idxes=idxes_sets[i])
        else: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, weights = weights)
        target = full_model.nf(full_model.delta(get_targets(full_model.predict_dynamics, batchvals)))

        # compute network values
        passive_prediction_params = full_model.passive_model(full_model.delta(batchvals.values.state))
        
        # compute losses
        done_flags = 1-batchvals.values.done
        passive_loss = - full_model.dist(*passive_prediction_params).log_prob(target)
        if full_model.multi_instanced: passive_loss = full_model.split_instances(passive_loss).sum(dim=2) * done_flags
        else: passive_loss = passive_loss.sum(dim=1).unsqueeze(1) * done_flags


        # optimize active and passive models
        prediction_params, active_loss = None, None
        if not train_args.no_pretrain_active:
            prediction_params = full_model.forward_model(full_model.gamma(batchvals.values.state))
            active_loss = - full_model.dist(*prediction_params).log_prob(target)
            if full_model.multi_instanced: active_loss = full_model.split_instances(active_loss).sum(dim=2) * done_flags
            else: active_loss = - full_model.dist(*prediction_params).log_prob(target).sum(dim=1).unsqueeze(1) * done_flags
            run_optimizer(active_optimizer, full_model.forward_model, active_loss)
        run_optimizer(passive_optimizer, full_model.passive_model, passive_loss)

        # only for unit testing
        if keep_outputs: outputs.append((prediction_params, passive_prediction_params))

        if i % train_args.log_interval == 0:
            _passive_logging(full_model, train_args, i, batchvals, target, passive_prediction_params, passive_loss, prediction_params, active_loss)
    return outputs