import os
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from test_network import TSNetworkWrapper

from Environments.environment_initializer import initialize_environment

from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen, AnglePolicy
from EnvironmentModels.Nav2D.Nav2D_environment_model import Nav2DEnvironmentModel
from Environments.Nav2D.Nav2D import Nav2D
from EnvironmentModels.Pushing.pushing_environment_model import PushingEnvironmentModel
from Environments.Pushing.screen import Pushing
from EnvironmentModels.Gym.gym_environment_model import GymEnvironmentModel
from Environments.RobosuitePushing.find_path import find_path

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.test_RL import testRL
from ReinforcementLearning.Policy.policy import TSPolicy, pytorch_model
from ReinforcementLearning.assess_policies import assess_policies
from EnvironmentModels.environment_model import FeatureSelector
from Options.Termination.termination import terminal_forms
from Options.done_model import DoneModel
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from Options.terminate_reward import TerminateReward
from DistributionalModels.InteractionModels.dummy_models import DummyBlockDatasetModel, DummyVariantBlockDatasetModel, DummyNegativeRewardDatasetModel, DummyMultiBlockDatasetModel, DummyStickDatasetModel
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model, interaction_models
from DistributionalModels.InteractionModels.samplers import samplers
from Rollouts.collector import OptionCollector
from Rollouts.param_buffer import ParamReplayBuffer

import tianshou as ts


if __name__ == '__main__':
    print("pid", os.getpid())

    args = get_args()
    torch.cuda.set_device(args.gpu)
    args.concatenate_param = True
    args.normalized_actions = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if len(args.record_rollouts) > 0 and args.render:
        render = "Test"
    elif len(args.visualize_param) > 0:
        render = "Param"
    else:
        render = ""
    environment, environment_model, args = initialize_environment(args, render=render)

    if args.true_environment:
        args.parameterized = args.env == "Nav2D"
    else:
        args.parameterized = True


    if args.dataset_dir != "dummy":
        if args.true_environment:
            dataset_model = interaction_models["dummy"](environment_model=environment_model)
        else:
            dataset_model = load_hypothesis_model(args.dataset_dir)
            torch.cuda.empty_cache()
            dataset_model.cpu()
        if len(args.force_mask):
            dataset_model.selection_binary = pytorch_model.wrap(args.force_mask, cuda = dataset_model.iscuda)
        dataset_model.environment_model = environment_model

    if args.object == "Block" and args.env == "SelfBreakout":
        args.num_instance = environment.num_blocks
        args.no_combine_param_mask = True
        if args.target_mode:
            dataset_model = DummyBlockDatasetModel(environment_model)
            dataset_model.environment_model = environment_model
        else:
            dataset_model = DummyMultiBlockDatasetModel(environment_model)
            dataset_model.environment_model = environment_model            
        dataset_model.sample_able.vals = np.array([dataset_model.sample_able.vals[0]]) # for some reason, there are some interaction values that are wrong
        discretize_actions = {0: np.array([-1,-1]), 1: np.array([-2,-1]), 2: np.array([-2,1]), 3: np.array([-1,1])}
    if args.object == "Reward" and args.env == "RoboPushing":
        dataset_model = DummyNegativeRewardDatasetModel(environment_model)
        dataset_model.environment_model = environment_model
        args.target_object = "Target"
        args.num_instance = args.num_obstacles
        args.reverse_feature_selector = dataset_model.cfnonselector[0]


    dataset_model.sample_continuous = True
    if args.sample_continuous != 0:
        dataset_model.sample_continuous = False if args.sample_continuous == 1 else True 
    if args.object == "Ball" and args.env == "SelfBreakout":
        dataset_model.sample_able.vals = [np.array([0,0,-1,-1,0]).astype(float), np.array([0,0,-2,-1,0]).astype(float), np.array([0,0,-2,1,0]).astype(float), np.array([0,0,-1,1,0]).astype(float)]
    dataset_model.environment_model = environment_model
    args.dataset_model = dataset_model
    init_state = environment.reset() # get an initial state to define shapes

    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule,
     environment_model=environment_model, init_state=init_state, no_combine_param_mask=args.no_combine_param_mask, 
     sample_distance=args.sample_distance, target_object=args.target, path_fn=find_path)
    if sampler is not None: sampler.dataset_model = dataset_model

    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    graph = load_graph(args.graph_dir)

    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = args.object in graph.nodes
    print(load_option, args.object)

    if args.change_option:
        load_option = graph.nodes[args.object].option
        load_option.cuda()
        print(load_option.policy.algo_policy.actor.last.model[0].weight.data, load_option.policy.algo_policy.actor.preprocess.model.model[0].weight.data)
        
        models = ObjDict()
        models.sampler = load_option.sampler
        models.state_extractor = load_option.state_extractor
        models.terminate_reward = load_option.terminate_reward
        models.action_map = load_option.action_map
        models.dataset_model = load_option.dataset_model
        models.temporal_extension_manager = load_option.temporal_extension_manager
        models.done_model = load_option.done_model
        next_option = load_option.next_option

        # initialize option
        option = option_forms[args.option_type](args, models, None, next_option)
        option.policy = load_option.policy
        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_map.mapped_action_shape)
        option.cpu()
        print(load_option.policy.algo_policy.actor.last.model[0].weight.data, option.policy.algo_policy.actor.preprocess.model.model[0].weight.data)
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
        option = graph.nodes[args.object].option
        # initialize new terminate_reward (for if we want to assess a different function than the one given)
        # initialize termination function, reward function, done model
        tt = args.terminal_type[:4] if args.terminal_type.find('inst') != -1 else args.terminal_type
        rt = args.reward_type[:4] if args.reward_type.find('inst') != -1 else args.reward_type
        termination = terminal_forms[tt](name=args.object, **vars(args))
        reward = reward_forms[rt](**vars(args)) # using the same epsilon for now, but that could change
        done_model = DoneModel(use_termination = args.use_termination, time_cutoff=args.time_cutoff, true_done_stopping = not args.not_true_done_stopping)

        # initialize terminate-reward
        args.reward = reward
        args.termination = termination
        args.state_extractor = option.state_extractor
        args.dataset_model = option.dataset_model
        terminate_reward = TerminateReward(args)

        option.terminate_reward = terminate_reward
        option.sampler = sampler
        if len(args.load_network) > 0:
            # data/breakout/network_test/net_ball_test_policy_small.pt
            network = torch.load(args.load_network)
            option.policy.algo_policy.model = TSNetworkWrapper(network)


    np.set_printoptions(threshold = 1000000, linewidth = 150, precision=3)


    policy = option.policy
    policy.option = option
    option.zero_epsilon()
    if args.epsilon > 0:
        option.set_epsilon(args.epsilon)
    option.sampler = sampler

    if args.cuda:
        option.cuda()
        option.set_device(args.gpu)
        dataset_model.cuda()
    graph.load_environment_model(environment_model)

    MAXEPISODELEN = 150
    test_collector = OptionCollector(option.policy, environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, test=True, args=args, environment_model=environment_model)

    # if args.set_time_cutoff:
    option.time_cutoff = -1
    if args.policy_type == "Assess":
        # TODO: bottom two lines are hardcoded
        policy = AnglePolicy(4)
        option.action_map.num_actions = 4
        assess_policies(args, test_collector, environment, environment_model, option, policy, names, graph)
    else:
        done_lengths = testRL(args, test_collector, environment, environment_model, option, names, graph)

    if len(args.save_graph) > 0:
        option.cpu()
        option.save(args.save_dir)
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
