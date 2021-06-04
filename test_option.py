import os
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

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
    elif args.env.find("Pushing") != -1:
        args.continuous = False
        environment = Pushing(pushgripper=True)
        if args.env == "StickPushing":
            environment = Pushing(pushgripper=False)
        environment.seed(args.seed)
        environment_model = PushingEnvironmentModel(environment)
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
        # dataset_model.control_min = [cfs.feature_range[0] for cfs in dataset_model.cfselectors]
        # dataset_model.control_max = [cfs.feature_range[1] for cfs in dataset_model.cfselectors]
        # HACKED ABOVE
    if len(args.force_mask) > 0:
        dataset_model.selection_binary = pytorch_model.wrap(np.array(args.force_mask),cuda=args.cuda)
    if args.sample_continuous != 0:
        dataset_model.sample_continuous = False if args.sample_continuous == 1 else True 

    # dataset_model = load_factored_model(args.dataset_dir)
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule)
    pr, models = ObjDict(), (dataset_model, environment_model, sampler) # policy_reward, featurizers
    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    graph = load_graph(args.graph_dir)
    # TODO: none of these are actually being used lol
    tt = args.terminal_type[:4] if args.terminal_type.find('inst') != -1 else args.terminal_type
    rt = args.reward_type[:4] if args.reward_type.find('inst') != -1 else args.reward_type
    ep, args.epsilon = args.epsilon, args.epsilon_close
    termination = terminal_forms[tt](name=args.object, dataset_model=dataset_model, environment=environment, **vars(args))
    print(rt, reward_forms[rt])
    reward = reward_forms[rt](dataset_model=dataset_model, interaction_minimum=dataset_model.interaction_minimum, environment=environment, **vars(args)) # using the same epsilon for now, but that could change
    args.epsilon, args.epsilon_close = ep, args.epsilon
    done_model = DoneModel(use_termination = args.use_termination, time_cutoff=args.time_cutoff, true_done_stopping = not args.not_true_done_stopping)
    print (dataset_model.name)

    # hack to fix old versions REMOVE
    action_option = graph.nodes["Action"].option
    action_option.action_featurizer = dataset_model.controllable[0]
    print(graph.nodes.keys())
    graph.nodes["Paddle"].option.param_first = False
    # graph.nodes["Ball"].option.param_first = False
    dataset_model.object_dim = 5
    dataset_model.multi_instanced = False
    # hacks above


    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = args.object in graph.nodes
    print(load_option, args.object)
    last_option = graph.nodes[args.object].option
    if args.change_option:
        pr.policy, pr.termination, pr.reward, pr.done_model = None, termination, reward, done_model
        pr.next_option = None if args.true_environment else graph.nodes[option_name].option
        pr.next_option = pr.next_option if not args.true_actions else graph.nodes["Action"].option # true actions forces the next option to be Action
        print("keys", list(graph.nodes.keys()))
        option = option_forms[args.option_type](pr, models, args.object, temp_ext=args.temporal_extend, relative_actions = args.relative_action, relative_state=args.relative_state, discretize_acts=args.discretize_actions, device=args.gpu) # TODO: make exploration noise more alterable 
        if args.object == "Action" or args.object == "Raw":
            option.discrete = not args.continuous
        else:
            option.discrete = False # assumes that all non-base options are continuous
        print(option_name, option.discrete)
        option.policy = last_option.policy
        option.policy.option = option
        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape)
    else:
        option.assign_models(models)
    option.termination = termination
    option.reward = reward # the reward function for this option

    np.set_printoptions(threshold = 1000000, linewidth = 150, precision=3)

    print("sample able", dataset_model.sample_able.vals)


    policy = option.policy
    if args.cuda:
        option.cuda()
        option.set_device(args.gpu)
        dataset_model.cuda()
    graph.load_environment_model(environment_model)

    MAXEPISODELEN = 150
    test_collector = OptionCollector(option.policy, environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, 
        use_param=args.parameterized, use_rel=args.relative_state, true_env=args.true_environment, test=True, 
        print_test=True, param_recycle = args.param_recycle)

    # if args.set_time_cutoff:
    option.time_cutoff = -1
    done_lengths = testRL(args, test_collector, environment, environment_model, option, names, graph)
    if len(args.save_graph) > 0:
        option.cpu()
        option.save(args.save_dir)
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
