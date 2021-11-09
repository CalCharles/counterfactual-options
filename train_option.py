import os, sys
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from Environments.environment_initializer import initialize_environment

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.train_RL import trainRL
from ReinforcementLearning.Policy.policy import TSPolicy, pytorch_model
from EnvironmentModels.environment_model import FeatureSelector, discretize_space
from Options.Termination.termination import terminal_forms
from Options.done_model import DoneModel
from Options.option_graph import OptionGraph, OptionNode, OptionEdge, load_graph
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from Options.action_map import PrimitiveActionMap, ActionMap
from Options.state_extractor import StateExtractor
from Options.terminate_reward import TerminateReward
from Options.temporal_extension_manager import TemporalExtensionManager
from DistributionalModels.distributional_model import load_factored_model
from DistributionalModels.InteractionModels.dummy_models import DummyBlockDatasetModel
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model, interaction_models
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

    print(type(args))

    # manage environment
    test_environment, test_environment_model, args = initialize_environment(args)
    environment, environment_model, args = initialize_environment(args)
    if args.true_environment:
        args.parameterized = args.env == "Nav2D"
    else:
        args.parameterized = True

    # initialize dataset model
    if args.true_environment:
        dataset_model = interaction_models["dummy"](environment_model=environment_model)
    else:
        dataset_model = load_hypothesis_model(args.dataset_dir)
        torch.cuda.empty_cache()
        dataset_model.cpu()
    if len(args.force_mask):
        dataset_model.selection_binary = pytorch_model.wrap(args.force_mask, cuda = dataset_model.iscuda)
    dataset_model.environment_model = environment_model
    init_state = environment.reset() # get an initial state to define shapes

    # block specific shortcuts
    discretize_actions = args.discretize_actions
    if args.object == "Block" and args.env == "SelfBreakout":
        if args.target_mode:
            dataset_model = DummyBlockDatasetModel(environment_model)
            dataset_model.environment_model = environment_model
        dataset_model.sample_able.vals = np.array([dataset_model.sample_able.vals[0]]) # for some reason, there are some interaction values that are wrong
        discretize_actions = {0: np.array([-1,-1]), 1: np.array([-2,-1]), 2: np.array([-2,1]), 3: np.array([-1,1])}

    if args.cuda: dataset_model.cuda()
    else: dataset_model.cpu()

    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        args.primitive_action_map = PrimitiveActionMap(args)
        args.action_featurizer = dataset_model.controllable[0] if environment.discrete_actions and not args.true_environment else dataset_model.controllable
        if environment.discrete_actions:
            actions = PrimitiveOption(args, None)
            nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
            graph = OptionGraph(nodes, dict(), dataset_model.controllable)
        else:
            actions = PrimitiveOption(args, None)
            nodes = {'Action': OptionNode('Action', actions, action_shape = environment.action_shape)}
            graph = OptionGraph(nodes, dict(), dataset_model.controllable)

    # TODO: REMOVE LINE BELOW
    # graph.nodes["Paddle"].option.terminate_reward.true_interaction = False
    graph.nodes["Action"].option.device = None
    if "Paddle" in graph.nodes: graph.nodes["Paddle"].option.state_extractor.use_pair_gamma = False
    if "Ball" in graph.nodes: graph.nodes["Ball"].option.state_extractor.use_pair_gamma = False
    if "Ball" in graph.nodes: graph.nodes["Ball"].option.state_extractor.hardcoded_normalization = ["breakout", 3, 1]
    if "Gripper" in graph.nodes: graph.nodes["Gripper"].option.state_extractor.use_pair_gamma = False

    # initialize sampler
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, sample_schedule=args.sample_schedule, environment_model=environment_model, init_state=init_state, no_combine_param_mask=args.no_combine_param_mask)

    # initialize state extractor
    option_name = dataset_model.name.split("->")[0] # TODO: assumes only one option in the tail for now
    args.object = "Reward" if len(args.object) == 0 else args.object
    args.name_pair = [option_name, args.object]
    next_option = None if args.true_environment else graph.nodes[option_name].option
    if args.primitive_actions:
        args.primitive_action_map = PrimitiveActionMap(args)
        args.action_featurizer = dataset_model.controllable[0] if environment.discrete_actions else [c for c in dataset_model.controllable if c.object() == "Action"]
        next_option = PrimitiveOption(args, None)

    if args.true_environment:
        option_selector = None
    else:
        option_selector = environment_model.create_entity_selector([option_name])

    full_state = environment.reset()
    args.dataset_model = dataset_model
    print(args.dataset_model)
    args.environment_model = environment_model
    args.action_feature_selector = next_option.dataset_model.feature_selector if next_option is not None and next_option.name != "Action" else environment_model.construct_action_selector()
    if sampler is None:
        param, mask = None, None
    else:
        param, mask = sampler.param, sampler.mask
    state_extractor = StateExtractor(args, option_selector, full_state, param, mask)

    # initialize termination function, reward function, done model
    tt = args.terminal_type[:4] if args.terminal_type.find('inst') != -1 else args.terminal_type
    rt = args.reward_type[:4] if args.reward_type.find('inst') != -1 else args.reward_type
    termination = terminal_forms[tt](name=args.object, **vars(args))
    reward = reward_forms[rt](**vars(args)) # using the same epsilon for now, but that could change
    done_model = DoneModel(use_termination = args.use_termination, time_cutoff=args.time_cutoff, true_done_stopping = not args.not_true_done_stopping)

    # initialize terminate-reward
    args.reward = reward
    args.termination = termination
    args.state_extractor = state_extractor
    terminate_reward = TerminateReward(args)

    if args.true_environment:
        args.discrete_actions = environment.discrete_actions
        action_map = PrimitiveActionMap(args)
    else:
        # initialize action_map
        args.discrete_actions = next_option.action_map.discrete_control is not None # 0 if continuous, also 0 if discretize_acts is used
        args.discretize_acts = discretize_actions if args.discretize_actions else None # none if not used
        args.num_actions = len(next_option.action_map.discrete_control) if next_option.action_map.discrete_control is not None else 0
        args.discrete_params = args.dataset_model.sample_able if args.sampler_type == "hst" else None# historical the only discrete sampler at the moment
        args.control_min, args.control_max = dataset_model.control_min, dataset_model.control_max# from dataset model
        action_map = ActionMap(args, next_option.action_map)

    # initialize temporal extension manager
    temporal_extension_manager = TemporalExtensionManager(args)

    # assign models
    models = ObjDict()
    models.sampler = sampler
    models.state_extractor = state_extractor
    models.terminate_reward = terminate_reward
    models.action_map = action_map
    models.dataset_model = dataset_model
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
        paction_space = action_map.policy_action_space # if converting continuous to discrete, otherwise the same
        num_inputs = int(np.prod(state_extractor.obs_shape))
        max_action = action_map.action_space.n if action_map.discrete_actions else 1 # might have problems with discretized actions
    else:
        action_space = environment.action_space
        paction_space = environment.action_space
        if args.policy_type in ['basic', 'point']:
            num_inputs = int(np.prod(state_extractor.obs_shape))
        else:
            num_inputs = environment.observation_space.shape

        max_action = environment.action_space.n if environment.discrete_actions else environment.action_space.high[0]
    option.time_cutoff = args.time_cutoff
    args.option = option
    args.first_obj_dim = state_extractor.first_obj_shape
    args.object_dim = state_extractor.object_shape + state_extractor.first_obj_shape
    if not load_option and not args.change_option:
        policy = TSPolicy(num_inputs, paction_space, action_space, max_action, **vars(args)) # default args?
    else:
        set_option = graph.nodes[args.object].option
        policy = set_option.policy
        policy.option = option

    action_map.assign_policy_map(policy.map_action, policy.reverse_map_action, policy.exploration_noise)

    if not load_option:
        option.policy = policy
        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = action_map.mapped_action_shape if not args.true_environment else environment.action_shape)
    else:
        graph.load_environment_model(environment_model)

    # apply cuda
    if args.cuda:
        option.cpu()
        option.cuda()
        option.set_device(args.gpu)
        policy.cuda()
        dataset_model.cuda()
    
    # debugging lines
    torch.set_printoptions(precision=2)
    np.set_printoptions(precision=3, linewidth = 150, threshold=200, suppress=True)
    
    # TODO: only initializes with ReplayBuffer, prioritizedReplayBuffer at the moment, but could extend to vector replay buffer if multithread possible
    if len(args.prioritized_replay) > 0:
        trainbuffer = ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])
    else:
        trainbuffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

    train_collector = OptionCollector(option.policy, environment, trainbuffer, exploration_noise=True, test=False,
                        option=option, environment_model = environment_model, args = args) # for now, no preprocess function
    MAXEPISODELEN = 100
    test_collector = OptionCollector(option.policy, test_environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, test=True, args=args, environment_model=test_environment_model)

    trained = trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph)
    if trained and not args.true_environment: # if trained, add control feature to the graph
        graph.cfs += dataset_model.cfselectors
        graph.add_edge(OptionEdge(args.object, option_name))
    if args.train and args.save_interval > 0:
        option.cpu()
        option.save(args.save_dir)
    if len(args.save_graph) > 0:
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], args.environment_model, cuda=args.cuda)
