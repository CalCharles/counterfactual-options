# primary train operator
import numpy as np
import os, cv2, time, copy
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Networks.network import ConstantNorm, pytorch_model
from tianshou.data import Collector, Batch, ReplayBuffer
from Environments.environment_initializer import initialize_environment
from Options.option_graph import OptionGraph, OptionNode, load_graph, OptionEdge, graph_construct_load
from DistributionalModels.InteractionModels.interaction_model import default_model_args, load_hypothesis_model

from DistributionalModels.InteractionModels.InteractionTraining.traces import get_proximal_indexes, generate_interaction_trace, adjust_interaction_trace
from DistributionalModels.InteractionModels.InteractionTraining.train_passive import train_passive
from DistributionalModels.InteractionModels.InteractionTraining.train_interaction import train_interaction
from DistributionalModels.InteractionModels.InteractionTraining.train_combined import train_combined
from DistributionalModels.InteractionModels.InteractionTraining.compute_errors import get_target_magnitude, get_prediction_error, get_error
from DistributionalModels.InteractionModels.InteractionTraining.train_utils import get_weights, run_optimizer, get_targets
from DistributionalModels.InteractionModels.InteractionTraining.init_interaction_network import init_model
from Networks.input_norm import InterInputNorm, PointwiseNorm
from EnvironmentModels.environment_normalization import hardcode_norm, position_mask

from Rollouts.rollouts import ObjDict, merge_rollouts



if __name__ == '__main__':
    '''
    Train the passive model, interaction model and active model
    @param control is the name of the object that we have control over
    @param controllers is the list of corresponding controllable feature selectors for object @param control 
    @param target_name is the name of the object that we want to control using @param control
    '''
    args = get_args()
    environment, environment_model, args = initialize_environment(args, set_save=False)
    graph, controllable_feature_selectors, args = graph_construct_load(args, environment, environment_model)

    model_args = default_model_args(args.predict_dynamics, args.policy_type) # input and output sizes should not be needed
    model_args.hidden_sizes, model_args.interaction_binary, model_args.interaction_prediction, model_args.init_form, model_args.activation, model_args.interaction_distance = args.hidden_sizes, args.interaction_binary, args.interaction_prediction, args.init_form, args.activation, args.interaction_distance
    model_args['controllable'], model_args['environment_model'] = controllable_feature_selectors, environment_model
    model_args.cuda=args.cuda
    cfsnames = [cfs.feature_selector.get_entity()[0] for cfs in controllable_feature_selectors]
    cfsdict = dict()
    cfslist = list()
    for cn, cfs in zip(cfsnames, controllable_feature_selectors):
        print(cn)
        if cn not in cfslist: cfslist.append(cn)
        if cn in cfsdict: cfsdict[cn].append(cfs)
        else: cfsdict[cn] = [copy.deepcopy(cfs)]

    cfs = "Paddle"
    control = cfs
    controllers = controllable_feature_selectors
    target = "Ball"
    target_name = target
    train = load_from_pickle("data/temp/train.pkl")
    test = load_from_pickle("data/temp/test.pkl")
    full_model = init_model(model_args, environment_model, cfs, cfsdict[cfs], list(), train, test, args, target)
    rollouts = train
    print("starting", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
    # define names
    full_model.control_feature = control # the name of the controllable object
    full_model.controllers = controllers
    control_name = full_model.control_feature
    full_model.target_name = target_name
    full_model.name = control + "->" + target_name
    train_args = args
    full_model.predict_dynamics = train_args.predict_dynamics

    # initialize the optimizers
    active_optimizer = optim.Adam(full_model.forward_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
    passive_optimizer = optim.Adam(full_model.passive_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
    interaction_optimizer = optim.Adam(full_model.interaction_model.parameters(), train_args.critic_lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
    
    # compute maximum and minimum of target values
    minmax = full_model.delta(rollouts.get_values('state'))
    full_model.control_min = np.amin(pytorch_model.unwrap(minmax), axis=1)
    full_model.control_max = np.amax(pytorch_model.unwrap(minmax), axis=1)

    # Computes the target normalization value, get normalization values
    if train_args.multi_instanced:
        output_norm_fun = PointwiseNorm(object_dim = full_model.object_dim)
    else:
        output_norm_fun = InterInputNorm()
    if len(train_args.hardcode_norm) > 0:
        if full_model.predict_dynamics:
            target_norm = hardcode_norm(train_args.hardcode_norm[0], ["Relative" + target_name])
            output_norm_fun.assign_mean_var(*target_norm)
        else:
            target_norm = hardcode_norm(train_args.hardcode_norm[0], [target_name])
            output_norm_fun.assign_mean_var(*target_norm)
    else:
        output_norm_fun.compute_input_norm(full_model.delta(get_targets(full_model.predict_dynamics, rollouts)))
    full_model.output_normalization_function = output_norm_fun
    full_model.inf = full_model.normalization_function
    full_model.irv = full_model.normalization_function.reverse
    full_model.dnf = full_model.delta_normalization_function
    full_model.drv = full_model.delta_normalization_function.reverse
    full_model.nf = full_model.output_normalization_function # temporarily to save length
    full_model.rv = full_model.output_normalization_function.reverse # same as above
    if train_args.cuda:
        full_model.inf.cuda()
        full_model.dnf.cuda()
        full_model.nf.cuda()

    # construct proximity batches if necessary
    non_proximal, proximal, non_proximal_weights = None, None, None
    if train_args.max_distance_epsilon > 0:
        proximal, non_proximal = get_proximal_indexes(full_model, train_args.position_mask, rollouts, train_args.max_distance_epsilon)
        non_proximal_weights = non_proximal / np.sum(non_proximal)


    # pre-initialize batches because it accelerates time
    batchvals = type(rollouts)(train_args.batch_size, rollouts.shapes)
    pbatchvals = type(rollouts)(train_args.batch_size, rollouts.shapes)
    half_batchvals = (type(rollouts)(train_args.batch_size // 2, rollouts.shapes), type(rollouts)(train_args.batch_size // 2, rollouts.shapes) ) # for the interaction model training 

    full_model.passive_model = torch.load("data/temp/passive_model.pt")
    full_model.passive_model.cpu()
    full_model.passive_model.cuda()

    passive_error_all = get_prediction_error(full_model, rollouts)
    # passive_error_all = full_model.interaction_model(full_model.gamma(rollouts.get_values("state")))
    # passive_error = pytorch_model.wrap(trace)
    weights, use_weights, total_live, total_dead, ratio_lambda = get_weights(passive_error_all, ratio_lambda = train_args.passive_weighting, 
        passive_error_cutoff=train_args.passive_error_cutoff, temporal_local=train_args.interaction_local, use_proximity=proximal)
    use_weights = (weights / np.sum(weights)).astype(np.float64)

    # handling boosting the passive operator to work with upweighted states
    # boosted_passive_operator = copy.deepcopy(full_model.passive_model)
    # true_passive = full_model.passive_model
    # full_model.passive_model = boosted_passive_operator
    passive_optimizer = optim.Adam(full_model.passive_model.parameters(), train_args.lr, eps=train_args.eps, betas=train_args.betas, weight_decay=train_args.weight_decay)
    print("combined", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))

    for i in range(train_args.num_iters):
        # passive failure weights
        idxes, batchvals = rollouts.get_batch(train_args.batch_size, weights=use_weights, existing=batchvals)
        prediction_params = full_model.forward_model(full_model.gamma(batchvals.values.state))
        passive_prediction_params = full_model.passive_model(full_model.delta(batchvals.values.state))
        target = full_model.output_normalization_function(pytorch_model.wrap(full_model.delta(get_targets(full_model.predict_dynamics, batchvals)), cuda=args.cuda))
        forward_log_probs = full_model.dist(*prediction_params).log_prob(target)
        forward_error = - forward_log_probs.sum(dim=1).unsqueeze(1)
        if i % 500 == 0:
            print(torch.cat([pytorch_model.wrap(full_model.delta(get_targets(full_model.predict_dynamics, batchvals)), cuda=args.cuda),
            full_model.rv(prediction_params[0]),
             prediction_params[1], passive_prediction_params[1], 
             forward_log_probs], dim = -1), forward_error.mean(), forward_log_probs.mean(dim=0))
        run_optimizer(active_optimizer, full_model.forward_model, forward_error)

