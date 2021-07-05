import os, sys
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from Environments.environment_initializer import initialize_environment

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.train_RL import trainRL
from ReinforcementLearning.Policy.policy import TSPolicy, pytorch_model
from EnvironmentModels.environment_model import FeatureSelector
from Options.Termination.termination import terminal_forms
from Options.done_model import DoneModel
from Options.option_graph import OptionGraph, OptionNode, OptionEdge, load_graph
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # manage environment
    environment, environment_model, args = initialize_environment(args)
    if args.true_environment:
        args.parameterized = args.env == "Nav2D"
    else:
        args.parameterized = True
        dataset_model.environment_model = environment_model

    # initialize dataset model
    if args.true_environment:
        dataset_model = interaction_models["dummy"](environment_model=environment_model)
    else:
        dataset_model = load_hypothesis_model(args.dataset_dir)
        torch.cuda.empty_cache()
        dataset_model.cpu()
    dataset_model.environment_model = environment_model
    # if args.object == "Block":
    #     dataset_model.sample_able.vals = np.array([dataset_model.sample_able.vals[0]]) # for some reason, there are some interaction values that are wrong
    #     args.discretize_actions = {0: np.array([-1,-1]), 1: np.array([-2,-1]), 2: np.array([-2,1]), 3: np.array([-1,1])}

    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule, environment_model=environment_model)
    if args.cuda: dataset_model.cuda()
    else: dataset_model.cpu()

    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        args.primitive_action_map = PrimitiveActionMap(args)
        if environment.discrete_actions:
            actions = PrimitiveOption(args, None)
            nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
            graph = OptionGraph(nodes, dict(), dataset_model.controllable)
        else:
            actions = PrimitiveOption(args, None)
            nodes = {'Action': OptionNode('Action', actions, action_shape = environment.action_shape)}
            graph = OptionGraph(nodes, dict(), dataset_model.controllable)
    
    # initialize state extractor
    option_name = dataset_model.name.split("->")[0]
    next_option = None if args.true_environment else graph.nodes[option_name].option
    full_state = environment.reset()
    args.dataset_model = dataset_model
    args.environment_model = environment_model
    args.action_feature_selector = next_option.dataset_model.feature_selector if next_option.name != "Action" else args.environment_model.action_selector
    state_extractor = StateExtractor(args, full_state)

    # initialize termination function, reward function, done model
    tt = args.terminal_type[:4] if args.terminal_type.find('inst') != -1 else args.terminal_type
    rt = args.reward_type[:4] if args.reward_type.find('inst') != -1 else args.reward_type
    termination = terminal_forms[tt](name=args.object, dataset_model=dataset_model, environment=environment, **vars(args))
    reward = reward_forms[rt](dataset_model=dataset_model, interaction_minimum=dataset_model.interaction_minimum, environment=environment, **vars(args)) # using the same epsilon for now, but that could change
    done_model = DoneModel(use_termination = args.use_termination, time_cutoff=args.time_cutoff, true_done_stopping = not args.not_true_done_stopping)
    
    # initialize terminate-reward
    args.reward = reward
    args.termination = termination
    args.state_extractor = args.state_extractor
    args.dataset_model = args.dataset_model
    terminate_reward = TerminateReward(args)

    # initialize action_map
    args.discrete_actions = next_option.discrete_control # 0 if continuous, also 0 if discretize_acts is used
    args.discretize_acts = environment_model.discretize_actions() # none if not used
    args.control_min, args.control_max = dataset_model.control_min, dataset_model.control_max# from dataset model
    action_map = ActionMap(args)

    # initialize temporal extension manager
    temporal_extension_manager = TemporalExtensionManager(args)

    # assign models
    models = ObjDict()
    models.sampler = sampler
    models.state_extractor = state_extractor
    models.terminate_reward = terminate_reward
    models.action_map = action_map
    models.temporal_extension_manager = temporal_extension_manager
    models.done_model = done_model

    # initialize option
    names = [args.object, option_name]
    load_option = not args.train and args.object in graph.nodes
    if not load_option or args.change_option:
        option = option_forms[args.option_type](args, models, None, next_option) # TODO: make exploration noise more alterable 
    else: # load the pre-constructed model
        option = graph.nodes[args.object].option
        self.assign_models(models)

    # initialize policy
    if not args.true_environment:
        action_space = action_map.action_space
        paction_space = option.policy_action_space # if converting continuous to discrete, otherwise the same
        num_inputs = int(np.prod(option.input_shape))
        max_action = option.action_max if option.discrete_actions else 1 # might have problems with discretized actions
    else:
        action_space = environment.action_space
        paction_space = environment.action_space
        num_inputs = environment.observation_space.shape
        max_action = environment.action_space.n if environment.discrete_actions else environment.action_space.high[0]
    option.time_cutoff = args.time_cutoff
    args.option = option
    args.first_obj_dim = option.first_obj_shape
    args.object_dim = option.object_shape + option.first_obj_shape
    if not load_option and not args.change_option:
        policy = TSPolicy(num_inputs, paction_space, action_space, max_action, discrete_actions=option.discrete_actions, **vars(args)) # default args?
    else:
        set_option = graph.nodes[args.object].option
        policy = set_option.policy
        policy.option = option
    action_map.assign_policy_map(policy.map_action, policy.reverse_map_action):
    if not load_option:
        option.policy = policy
        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape)
    else:
        graph.load_environment_model(environment_model)

    # apply cuda
    if args.cuda:
        option.cuda()
        option.set_device(args.gpu)
        policy.cuda()
        dataset_model.cuda()
    
    # debugging lines
    torch.set_printoptions(precision=2)
    np.set_printoptions(precision=2, linewidth = 150, threshold=200)
    
    # TODO: only initializes with ReplayBuffer, prioritizedReplayBuffer at the moment, but could extend to vector replay buffer if multithread possible
    if len(args.prioritized_replay) > 0:
        trainbuffer = ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])
    else:
        trainbuffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

    train_collector = OptionCollector(option.policy, environment, trainbuffer, exploration_noise=True, 
                        option=option, use_param=args.parameterized, use_rel=args.relative_state, 
                        true_env=args.true_environment, param_recycle=args.param_recycle) # for now, no preprocess function
    MAXEPISODELEN = 100
    test_collector = OptionCollector(option.policy, environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, use_param=args.parameterized, use_rel=args.relative_state, true_env=args.true_environment, test=True, print_test=args.print_test, grayscale=args.grayscale)

    trained = trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph)
    if trained and not args.true_environment: # if trained, add control feature to the graph
        graph.cfs += dataset_model.cfselectors
        graph.add_edge(OptionEdge(args.object, option_name))
    if args.train and args.save_interval > 0:
        option.cpu()
        option.save(args.save_dir)
    if len(args.save_graph) > 0:
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], cuda=args.cuda)