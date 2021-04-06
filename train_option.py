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
from ReinforcementLearning.train_RL import trainRL
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

import tianshou as ts

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    args.concatenate_param = True
    args.normalized_actions = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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


    # extra line for older dataset models
    # dataset_model.sample_continuous = False
    # dataset_model.use_layer_norm = False
    # if not args.true_environment:
    #     dataset_model.interaction_model.use_layer_norm = False
    # REMOVE above

    # reduced state HACK
    # dataset_model.gamma = FeatureSelector([6], {"Paddle": 1})
    # dataset_model.delta = FeatureSelector([6], {"Paddle": 1})
    # dataset_model.selection_binary = torch.ones((1)).cuda()
    # dataset_model.selection_binary = torch.tensor([0,0,1,1,0])
    # REMOVE above once alternative selection implemented

    # print(dataset_model.selection_binary)
    # dataset_model = load_factored_model(args.dataset_dir)
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule)
    pr, models = ObjDict(), (dataset_model, environment_model, sampler) # policy_reward, featurizers
    if args.cuda:
        dataset_model.cuda()
    try:
        graph = load_graph(args.graph_dir, args.buffer_len)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, models, "Action")
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
        graph = OptionGraph(nodes, dict(), dataset_model.controllable)
    termination = terminal_forms[args.terminal_type](name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close, interaction_probability=args.interaction_probability)
    reward = reward_forms[args.reward_type](epsilon=args.epsilon_close, parameterized_lambda=args.parameterized_lambda, reward_constant= args.reward_constant, interaction_model=dataset_model.interaction_model, interaction_minimum=dataset_model.interaction_minimum) # using the same epsilon for now, but that could change
    print (dataset_model.name)
    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = not args.train and args.object in graph.nodes
    print(load_option, option_name, args.object)

    # hack to fix old versions REMOVE
    # graph.nodes[option_name].option.output_prob_shape = (graph.nodes[option_name].option.dataset_model.delta.output_size(), )
    
    if not load_option:
        pr.policy, pr.termination, pr.reward, pr.next_option = None, termination, reward, (None if args.true_environment else graph.nodes[option_name].option)
        print("keys", list(graph.nodes.keys()))
        option = option_forms[args.option_type](pr, models, args.object, temp_ext=False) # TODO: make exploration noise more alterable 
        if option_name == "Action" or args.object == "Raw":
            option.discrete = not args.continuous
        else:
            option.discrete = False # assumes that all non-base options are continuous
    else:
        option = graph.nodes[args.object].option
        self.assign_models(models)

    if not args.true_environment:
        action_space = option.action_space
        num_inputs = int(np.prod(option.input_shape))
        max_action = option.action_max
    else:
        action_space = environment.action_space
        num_inputs = environment.observation_space.shape
        max_action = environment.action_space.high[0]
    option.time_cutoff = args.time_cutoff

    if not load_option:
        policy = TSPolicy(num_inputs, action_space, max_action, **vars(args)) # default args?
    else:
        policy = option.policy

    if args.cuda:
        policy.cuda()
        option.cuda()
        dataset_model.cuda()
    if not load_option:
        option.policy = policy

        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape)
    else:
        graph.load_environment_model(environment_model)
    
    # TODO: only initializes with ReplayBuffer at the moment, but could extend to prioritized replay buffer, vector replay buffer if multithread possible
    train_collector = OptionCollector(option.policy, environment, ts.data.ReplayBuffer(args.buffer_len, 1), exploration_noise=True, option=option) # for now, no preprocess function
    test_collector = OptionCollector(option.policy, environment, exploration_noise=True, option=option)

    done_lengths, trained = trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph)
    
    # done_lengths = np.array(done_lengths)
    # time_cutoff = 100
    # if len(done_lengths) > 0:
    #     time_cutoff = np.round_(np.quantile(done_lengths, .9))
    # if trained: # if trained, add control feature to the graph
    #     graph.cfs += dataset_model.cfselectors

    # if args.train and args.save_interval > 0:
    #     option.save(args.save_dir)
    # if args.set_time_cutoff:
    #     option.time_cutoff = time_cutoff
    # if len(args.save_graph) > 0:
    #     print(args.object, graph.nodes[args.object].option.time_cutoff)
    #     graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
    # print (time_cutoff)
