import os
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from Environments.environment_initializer import initialize_environment

from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D
from EnvironmentModels.Pushing.pushing_environment_model import PushingEnvironmentModel
from Environments.Pushing.screen import Pushing
from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.test_RL import testRL
from ReinforcementLearning.Policy.policy import TSPolicy, pytorch_model
from EnvironmentModels.environment_model import FeatureSelector
from Options.Termination.termination import terminal_forms
from Options.done_model import DoneModel
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model, interaction_models
from DistributionalModels.distributional_model import load_factored_model
from DistributionalModels.InteractionModels.samplers import samplers
from Rollouts.collector import OptionCollector
from Rollouts.param_buffer import ParamReplayBuffer

import tianshou as ts


if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    args.concatenate_param = True
    args.normalized_actions = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    environment, environment_model, args = initialize_environment(args)

    if args.true_environment:
        args.parameterized = args.env == "Nav2D"
    else:
        args.parameterized = True

    if args.true_environment:
        dataset_model = interaction_models["dummy"](environment_model=environment_model)
    else:
        dataset_model = load_hypothesis_model(args.dataset_dir)
        torch.cuda.empty_cache()
        dataset_model.cpu()
    if len(args.force_mask) > 0:
        dataset_model.selection_binary = pytorch_model.wrap(np.array(args.force_mask),cuda=args.cuda)
    if args.sample_continuous != 0:
        dataset_model.sample_continuous = False if args.sample_continuous == 1 else True 

    dataset_model.environment_model = environment_model
    init_state = environment.reset() # get an initial state to define shapes

    # dataset_model = load_factored_model(args.dataset_dir)
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule, environment_model=environment_model, init_state=init_state, no_combine_param_mask=args.no_combine_param_mask)


    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    graph = load_graph(args.graph_dir)

    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = args.object in graph.nodes
    print(load_option, args.object)
    last_option = graph.nodes[args.object].option
    if args.change_option:
        pass
        # TODO: make this work
        # pr.policy, pr.termination, pr.reward, pr.done_model = None, termination, reward, done_model
        # pr.next_option = None if args.true_environment else graph.nodes[option_name].option
        # pr.next_option = pr.next_option if not args.true_actions else graph.nodes["Action"].option # true actions forces the next option to be Action
        # print("keys", list(graph.nodes.keys()))
        # option = option_forms[args.option_type](pr, models, args.object, temp_ext=args.temporal_extend, relative_actions = args.relative_action, relative_state=args.relative_state, discretize_acts=args.discretize_actions, device=args.gpu) # TODO: make exploration noise more alterable 
        # if args.object == "Action" or args.object == "Raw":
        #     option.discrete = not args.continuous
        # else:
        #     option.discrete = False # assumes that all non-base options are continuous
        # print(option_name, option.discrete)
        # option.policy = last_option.policy
        # option.policy.option = option
        # graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape)
    else:
        option = last_option

    np.set_printoptions(threshold = 1000000, linewidth = 150, precision=3)

    print("sample able", dataset_model.sample_able.vals)


    policy = option.policy
    if args.cuda:
        option.cuda()
        option.set_device(args.gpu)
        dataset_model.cuda()
    graph.load_environment_model(environment_model)

    MAXEPISODELEN = 150
    test_collector = OptionCollector(option.policy, environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, test=True, args=args)

    # if args.set_time_cutoff:
    option.time_cutoff = -1
    done_lengths = testRL(args, test_collector, environment, environment_model, option, names, graph)
    if len(args.save_graph) > 0:
        option.cpu()
        option.save(args.save_dir)
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
