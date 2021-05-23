import os, sys
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
from ReinforcementLearning.train_RL import trainRL
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
from Rollouts.param_buffer import ParamReplayBuffer, ParamPriorityReplayBuffer

import tianshou as ts

if __name__ == '__main__':
    print(sys.argv)
    args = get_args()
    torch.cuda.set_device(args.gpu)
    args.concatenate_param = True
    args.normalized_actions = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.preprocess = None
    if args.env == "SelfBreakout":
        args.continuous = False
        environment = Screen(drop_stopping=args.drop_stopping)
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
        torch.cuda.empty_cache()
        dataset_model.cpu()
        print(dataset_model.interaction_model.model[0].weight.data, dataset_model.passive_model.mean.model[0].weight.data)
        dataset_model.environment_model = environment_model
        # HACKED BELOW
        # dataset_model.control_min = [cfs.feature_range[0] for cfs in dataset_model.cfselectors]
        # dataset_model.control_max = [cfs.feature_range[1] for cfs in dataset_model.cfselectors]
        # HACKED ABOVE

    # hacked forced velocity mask
    if len(args.force_mask) > 0:
        dataset_model.selection_binary = pytorch_model.wrap(np.array(args.force_mask),cuda=args.cuda)
    if args.sample_continuous != 0:
        dataset_model.sample_continuous = False if args.sample_continuous == 1 else True 
    # dataset_model.selection_binary[0] = 0
    # dataset_model.selection_binary[1] = 0
    # hacked forced location mask
    # dataset_model.selection_binary[2] = 0
    # dataset_model.selection_binary[3] = 0
    # dataset_model.sample_continuous = True

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
    if args.cuda: dataset_model.cuda()
    else: dataset_model.cpu()
    print(dataset_model.iscuda, dataset_model.interaction_model.model[0].weight.data)
    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, models, "Action", action_featurizer = dataset_model.controllable[0])
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
        graph = OptionGraph(nodes, dict(), dataset_model.controllable)
    termination = terminal_forms[args.terminal_type](name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close, 
                                                    interaction_probability=args.interaction_probability, env=environment, param_interaction=args.param_interaction)
    reward = reward_forms[args.reward_type](epsilon=args.epsilon_close, parameterized_lambda=args.parameterized_lambda, interaction_lambda = args.interaction_lambda, 
                                            reward_constant= args.reward_constant, interaction_model=dataset_model.interaction_model, interaction_minimum=dataset_model.interaction_minimum,
                                            env=environment, true_reward_lambda=args.true_reward_lambda) # using the same epsilon for now, but that could change
    done_model = DoneModel(use_termination = args.use_termination, time_cutoff=args.time_cutoff, true_done_stopping = not args.not_true_done_stopping)
    print (dataset_model.name)
    option_name = dataset_model.name.split("->")[0]
    names = [args.object, option_name]
    load_option = not args.train and args.object in graph.nodes
    print(load_option, option_name, args.object)

    # hack to fix old versions REMOVE
    action_option = graph.nodes["Action"].option
    action_option.action_featurizer = dataset_model.controllable[0]
    # graph.nodes["Action"].option.terminated = True
    # graph.nodes["Paddle"].option.discretize_actions = False
    # graph.nodes[option_name].option.output_prob_shape = (graph.nodes[option_name].option.dataset_model.delta.output_size(), )
    
    if not load_option:
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
    else:
        option = graph.nodes[args.object].option
        self.assign_models(models)

    if not args.true_environment:
        action_space = option.action_space if args.relative_action < 0 else option.relative_action_space
        paction_space = option.policy_action_space if args.discretize_actions else action_space # if converting continuous to discrete, otherwise the same
        num_inputs = int(np.prod(option.input_shape))
        max_action = option.action_max if option.discrete_actions else 1 # might have problems with discretized actions
    else:
        action_space = environment.action_space
        paction_space = environment.action_space
        num_inputs = environment.observation_space.shape
        max_action = environment.action_space.n if environment.discrete_actions else environment.action_space.high[0]
    option.time_cutoff = args.time_cutoff

    args.option = option
    if not load_option:
        policy = TSPolicy(num_inputs, paction_space, action_space, max_action, discrete_actions=option.discrete_actions, **vars(args)) # default args?
        if args.true_environment and args.env == "Nav2D": option.param_process = environment.param_process
        # policy.algo_policy.load_state_dict(torch.load("data/TSTestPolicy.pt"))
    else:
        policy = option.policy
        policy.option = policy

    if args.cuda:
        option.cuda()
        option.set_device(args.gpu)
        policy.cuda()
        dataset_model.cuda()
    if not load_option:
        option.policy = policy

        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape)
    else:
        graph.load_environment_model(environment_model)
    
    # debugging lines
    torch.set_printoptions(precision=2)
    np.set_printoptions(precision=2)
    
    # TODO: only initializes with ReplayBuffer, prioritizedReplayBuffer at the moment, but could extend to vector replay buffer if multithread possible
    if len(args.prioritized_replay) > 0:
        trainbuffer = ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])
    else:
        trainbuffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

    train_collector = OptionCollector(option.policy, environment, trainbuffer, exploration_noise=True, 
                        option=option, use_param=args.parameterized, use_rel=args.relative_state, 
                        true_env=args.true_environment, param_recycle=args.param_recycle) # for now, no preprocess function
    MAXEPISODELEN = 100
    test_collector = OptionCollector(option.policy, environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, use_param=args.parameterized, use_rel=args.relative_state, true_env=args.true_environment, test=True, print_test=args.print_test)
    # test_collector = ts.data.Collector(option.policy, environment)
    print("Check option discrete", option.object_name, option.discrete)
    trained = trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph)
    if trained and not args.true_environment: # if trained, add control feature to the graph
        graph.cfs += dataset_model.cfselectors
    if args.train and args.save_interval > 0:
        option.cpu()
        option.save(args.save_dir)
    if len(args.save_graph) > 0:
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)
    
    # done_lengths = np.array(done_lengths)
    # time_cutoff = 100
    # if len(done_lengths) > 0:
    #     time_cutoff = np.round_(np.quantile(done_lengths, .9))
    # if args.set_time_cutoff:
    #     option.time_cutoff = time_cutoff
    # print (time_cutoff)
