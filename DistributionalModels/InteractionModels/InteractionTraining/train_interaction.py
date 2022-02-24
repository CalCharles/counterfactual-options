# train interaction directly
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
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import run_optimizer, get_weights, get_targets

def _interaction_logging(full_model, train_args, batchvals, i, trace_loss, trw, trace, interaction_likelihood, target):
    obj_indices = list(range(min(30, train_args.batch_size)))
    inp = full_model.gamma(batchvals.values.state)
    if full_model.multi_instanced: 
        # print out only the interaction instances which are true
        # TODO: a ton of logging code that I no longer understand
        # target = target
        inp = full_model.split_instances(inp, full_model.object_dim)
        obj_indices = pytorch_model.unwrap((trace[idxes] > 0).nonzero())
        objective = full_model.delta(get_targets(full_model.predict_dynamics, batchvals))
        all_indices = []
        for ti in obj_indices:
            all_indices.append(np.array([ti[0], ti[1]-2]))
            all_indices.append(np.array([ti[0], ti[1]-1]))
            all_indices.append(pytorch_model.unwrap(ti))
            if ti[1]+1 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+1]))
            if ti[1]+2 < interaction_likelihood[ti[0]].shape[0]: all_indices.append(np.array([ti[0], ti[1]+2]))
            for i in range(3):
                all_indices.append(np.array([ti[0], np.random.randint(interaction_likelihood[ti[0]].shape[0])]))
        for _ in range(20):
            all_indices.append(np.array([np.random.randint(train_args.batch_size), np.random.randint(interaction_likelihood.shape[1])]))
        obj_indices = np.array(all_indices)
        # print (obj_indices)
        print("Iters: ", i, ": tl: ", trace_loss)
        # print(target.shape)
        print(target[obj_indices[0][0]], interaction_likelihood[obj_indices[0][0]])
        for a in obj_indices:
            print("state:", pytorch_model.unwrap(inp)[a[0], a[1]],
            "inter: ", pytorch_model.unwrap(interaction_likelihood[a[0], a[1]]),
            "target: ", pytorch_model.unwrap(target[a[0], a[1]]))
    else:
        print("Iters: ", i, ": tl: ", trace_loss)
        print("\nstate:", pytorch_model.unwrap(inp)[obj_indices],
            "\ntraining: ", pytorch_model.unwrap(interaction_likelihood[obj_indices]),
            "\ntarget: ", pytorch_model.unwrap(target[obj_indices]))


def train_interaction(full_model, rollouts, train_args, batchvals, trace, trace_targets, interaction_optimizer, idxes_sets=None, keep_outputs=False, weights=None):
    outputs = list()
    inter_loss = nn.BCELoss()
    if train_args.interaction_iters > 0:
        # in the multi-instanced case, if ANY interaction occurs, we want to upweight that state
        if weights is None:
            trw = torch.max(trace, dim=1)[0].squeeze() if full_model.multi_instanced else trace
        else:
            trw = weights
        print(trw.sum())
        # weights the values
        _, weights, live, dead, ratio_lambda = get_weights(ratio_lambda=train_args.interaction_weight, weights=trw, temporal_local=train_args.interaction_local)
        for i in range(train_args.interaction_iters):
            # get the input and target values
            if idxes_sets is not None: idxes, batchvals = rollouts.get_batch(train_args.batch_size, existing=batchvals, idxes=idxes_sets[i])
            else: idxes, batchvals = rollouts.get_batch(train_args.batch_size, weights=pytorch_model.unwrap(weights), existing=batchvals)
            target = trace_targets[idxes]# if full_model.multi_instanced else trace[idxes]
            
            # get the network outputs
            # multi-instanced will have shape [batch, num_instances]
            if full_model.multi_instanced: interaction_likelihood = full_model.interaction_model.instance_labels(full_model.gamma(batchvals.values.state))
            else: interaction_likelihood = full_model.interaction_model(full_model.gamma(batchvals.values.state))
            
            # keep outputs only for unit testing
            if keep_outputs: outputs.append(interaction_likelihood)

            # compute loss
            trace_loss = inter_loss(interaction_likelihood.squeeze(), target)
            run_optimizer(interaction_optimizer, full_model.interaction_model, trace_loss)
            
            if i % train_args.log_interval == 0:
                _interaction_logging(full_model, train_args, batchvals, i, trace_loss, trw, trace, interaction_likelihood, target)
                weight_lambda = train_args.interaction_weight * np.exp(-i/(train_args.interaction_iters) * 5)
                _, weights, live, dead, ratio_lambda = get_weights(ratio_lambda=weight_lambda, weights=trw, temporal_local=train_args.interaction_local)
        full_model.interaction_model.needs_grad=False # no gradient will pass through the interaction model
        # if train_args.save_intermediate:
        #     torch.save(full_model.interaction_model, "data/temp/interaction_model.pt")
    # if train_args.load_intermediate:
    #     full_model.interaction_model = torch.load("data/temp/interaction_model.pt")
    return outputs