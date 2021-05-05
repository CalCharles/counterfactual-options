import os
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D
from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.test_RL import testRL
from ReinforcementLearning.Policy.policy import TSPolicy, pytorch_model
from EnvironmentModels.environment_model import FeatureSelector
from Options.Termination.termination import terminal_forms
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
    args.preprocess = None
    if args.env == "SelfBreakout":
        args.continuous = False
        environment = Screen()
        environment.seed(args.seed)
        environment_model = BreakoutEnvironmentModel(environment)
    elif args.env == "Nav2D":
        args.continuous = False
        environment = Nav2D()
        environment.seed(args.seed)
        environment_model = Nav2DEnvironmentModel(environment)
        if args.true_environment:
            args.preprocess = environment.preprocess
    elif args.env[:6] == "gymenv":
        args.continuous = True
        from Environments.Gym.gym import Gym
        environment = Gym(gym_name= args.env[6:])
        environment.seed(args.seed)
        environment_model = GymEnvironmentModel(environment)
        args.normalized_actions = True
    environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
    if args.true_environment:
        args.parameterized = args.env == "Nav2D"
        args.concatenate_param = False
        dataset_model = interaction_models["dummy"](environment_model=environment_model)
    else:
        args.parameterized = True
        dataset_model = load_hypothesis_model(args.dataset_dir)
        dataset_model.environment_model = environment_model
        # HACKED BELOW
        dataset_model.control_min = [cfs.feature_range[0] for cfs in dataset_model.cfselectors]
        dataset_model.control_max = [cfs.feature_range[1] for cfs in dataset_model.cfselectors]
        # HACKED ABOVE

    # dataset_model = load_factored_model(args.dataset_dir)
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule)
    pr, models = ObjDict(), (dataset_model, environment_model, sampler) # policy_reward, featurizers
    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    graph = load_graph(args.graph_dir)
    termination = terminal_forms[args.terminal_type](name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close, interaction_probability=args.interaction_probability, env=environment)
    reward = reward_forms[args.reward_type](epsilon=args.epsilon_close, parameterized_lambda=args.parameterized_lambda, reward_constant= args.reward_constant, interaction_model=dataset_model.interaction_model, interaction_minimum=dataset_model.interaction_minimum, env=environment) # using the same epsilon for now, but that could change
    print (dataset_model.name)
    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = args.object in graph.nodes
    print(load_option, args.object)
    option = graph.nodes[args.object].option
    option.assign_models(models)
    option.termination = termination
    option.reward = reward # the reward function for this option


    policy = option.policy
    if args.cuda:
        policy.cuda()
        option.cuda()
        dataset_model.cuda()
    graph.load_environment_model(environment_model)

    MAXEPISODELEN = 150
    test_collector = OptionCollector(option.policy, environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, use_param=args.parameterized, use_rel=args.relative_state, true_env=args.true_environment, test=True, print_test=True)

    # if args.set_time_cutoff:
    # option.time_cutoff = -1
    done_lengths = testRL(args, test_collector, environment, environment_model, option, names, graph)
