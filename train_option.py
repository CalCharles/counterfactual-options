import os, sys, psutil
import torch
import numpy as np
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle

from Environments.environment_initializer import initialize_environment

from Rollouts.rollouts import ObjDict
from ReinforcementLearning.train_RL import trainRL
from ReinforcementLearning.Policy.loaded_policy import LoadedPolicy
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
from DistributionalModels.InteractionModels.dummy_models import DummyBlockDatasetModel, DummyVariantBlockDatasetModel,DummyVariantBlockHeightDatasetModel, DummyNegativeRewardDatasetModel, DummyMultiBlockDatasetModel, DummyStickDatasetModel
from DistributionalModels.InteractionModels.interaction_model import load_hypothesis_model, interaction_models
from DistributionalModels.InteractionModels.samplers import samplers
from Rollouts.collector import OptionCollector
from Rollouts.param_buffer import ParamReplayBuffer, ParamPriorityReplayBuffer
from torch.utils.tensorboard import SummaryWriter


import tianshou as ts

if __name__ == '__main__':
    print("pid", os.getpid())
    print(" ".join(sys.argv))
    args = get_args()
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(type(args))

    # manage environment
    if len(args.record_rollouts) > 0 and args.render:
        render = "Test"
    elif len(args.visualize_param) > 0:
        render = "Param"
    else:
        render = ""
    test_environment, test_environment_model, args = initialize_environment(args, set_save=len(args.record_rollouts) != 0, render=render)
    environment, environment_model, args = initialize_environment(args, set_save=len(args.record_rollouts) != 0, render=render if args.visualize_param else "")
    if args.true_environment:
        args.parameterized = args.env == "Nav2D"
    else:
        args.parameterized = True

    # initialize dataset model
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
    init_state = environment.reset() # get an initial state to define shapes

    # block specific shortcuts
    discretize_actions = args.discretize_actions
    target_object = "Reward"
    args.num_instance = 1
    args.target_instanced=False
    args.confirm_interaction = False
    if args.object == "Ball" and args.env == "SelfBreakout":
        args.confirm_interaction = True
        dataset_model.interaction_prediction = 0.05
        dataset_model.sample_able.vals = [np.array([0,0,-1,-1,0]).astype(float), np.array([0,0,-2,-1,0]).astype(float), np.array([0,0,-2,1,0]).astype(float), np.array([0,0,-1,1,0]).astype(float)]
    if args.object == "Block" and args.env == "SelfBreakout":
        args.num_instance = environment.num_blocks
        args.target_instanced = True
        args.no_combine_param_mask = True
        if args.target_mode:
            dataset_model = DummyBlockDatasetModel(environment_model)
            dataset_model.environment_model = environment_model
        else:
            dataset_model = DummyMultiBlockDatasetModel(environment_model)
            dataset_model.environment_model = environment_model            
        dataset_model.sample_able.vals = np.array([dataset_model.sample_able.vals[0]]) # for some reason, there are some interaction values that are wrong
        discretize_actions = {0: np.array([-1,-1]), 1: np.array([-2,-1]), 2: np.array([-2,1]), 3: np.array([-1,1])}
    if args.object == "Reward" and args.env == "SelfBreakout":
        args.num_instance = environment.num_blocks
        args.target_instanced = True
        args.no_combine_param_mask = True
        dataset_model = DummyVariantBlockDatasetModel(environment_model)
        # dataset_model = DummyVariantBlockHeightDatasetModel(environment_model)
        dataset_model.environment_model = environment_model
        dataset_model.sample_able.vals = np.array([dataset_model.sample_able.vals[0]]) # for some reason, there are some interaction values that are wrong
        discretize_actions = {0: np.array([-1,-1]), 1: np.array([-2,-1]), 2: np.array([-2,1]), 3: np.array([-1,1])}
        args.object = "Block" # switch the object to block for parameter 
    if args.object == "Reward" and args.env == "RoboPushing":
        dataset_model = DummyNegativeRewardDatasetModel(environment_model)
        dataset_model.environment_model = environment_model
        args.target_object = "Target"
        args.num_instance = args.num_obstacles
        args.reverse_feature_selector = dataset_model.cfnonselector[0]
    if args.object == "Stick" and args.env == "RoboStick":
        dataset_model = DummyStickDatasetModel(environment_model)
        dataset_model.environment_model = environment_model
    if args.env == "RoboPushing":
        args.num_instance = args.num_obstacles

    if args.cuda: dataset_model.cuda()
    else: dataset_model.cpu()

    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        args.primitive_action_map = PrimitiveActionMap(args)
        args.action_featurizer = dataset_model.controllable[0] if environment.discrete_actions else dataset_model.controllable
        if environment.discrete_actions:
            actions = PrimitiveOption(args, None)
            nodes = {'Action': OptionNode('Action', actions, action_shape = (1,))}
            graph = OptionGraph(nodes, dict(), dataset_model.controllable)
        else:
            actions = PrimitiveOption(args, None)
            nodes = {'Action': OptionNode('Action', actions, action_shape = environment.action_shape)}
            graph = OptionGraph(nodes, dict(), dataset_model.controllable)
    
    # initialize sampler
    sampler = None if args.true_environment else samplers[args.sampler_type](dataset_model=dataset_model, 
        sample_schedule=args.sample_schedule, environment_model=environment_model, init_state=init_state, 
        no_combine_param_mask=args.no_combine_param_mask, sample_distance=args.sample_distance, target_object=args.target)

    # add in the sampler to the environments if necessary
    if args.object == "Block" and args.env == "SelfBreakout" and args.breakout_variant == "proximity":
        test_environment.sampler = samplers[args.sampler_type](dataset_model=dataset_model, 
            sample_schedule=args.sample_schedule, environment_model=test_environment_model, init_state=init_state, 
            no_combine_param_mask=args.no_combine_param_mask, sample_distance=args.sample_distance, target_object=args.target)
        environment.sampler = samplers[args.sampler_type](dataset_model=dataset_model, 
            sample_schedule=args.sample_schedule, environment_model=environment_model, init_state=init_state, 
            no_combine_param_mask=args.no_combine_param_mask, sample_distance=args.sample_distance, target_object=args.target)

    # initialize state extractor
    option_name = dataset_model.name.split("->")[0].split("+")[0] # TODO: assumes only one option in the tail for now
    full_tail = dataset_model.name.split("->")[0].split("+")
    args.object = "Reward" if len(args.object) == 0 else args.object
    args.name_pair = [option_name, args.object]
    next_option = None if args.true_environment else graph.nodes[option_name].option
    if args.primitive_actions:
        args.primitive_action_map = PrimitiveActionMap(args)
        args.action_featurizer = dataset_model.controllable[0] if environment.discrete_actions else [c for c in dataset_model.controllable if c.object() == "Action"]
        next_option = PrimitiveOption(args, None)
    option_selector = environment_model.create_entity_selector([option_name]) # TODO: should be full_tail probably 
    full_state = environment.reset()
    args.dataset_model = dataset_model
    args.environment_model = environment_model
    args.action_feature_selector = next_option.dataset_model.feature_selector if next_option is not None and next_option.name != "Action" else environment_model.construct_action_selector()
    # args.action_reverse_selector = next_option.dataset_model.reverse_feature_selector if next_option is not None and next_option.name != "Action" else None
    if sampler is None: param, mask = None, None
    else: param, mask = sampler.param, sampler.mask
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

    # initialize action_map
    args.discrete_actions = next_option.action_map.discrete_control is not None # 0 if continuous, also 0 if discretize_acts is used
    args.discretize_acts = discretize_actions if args.discretize_actions else None # none if not used
    args.num_actions = len(next_option.action_map.discrete_control) if next_option.action_map.discrete_control is not None else 0
    args.discrete_params = args.dataset_model.sample_able if args.sampler_type == "hst" else None# historical the only discrete sampler at the moment
    args.control_min, args.control_max = dataset_model.control_min, dataset_model.control_max# from dataset model
    if args.object == "Reward" and args.env == "RoboPushing":
        next_option.action_map.control_max = np.array([.07, .17])
        next_option.action_map.control_min = np.array([-.26, -.17])
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
        option.assign_models(models)

    # initialize policy
    if not args.true_environment:
        action_space = action_map.action_space
        paction_space = action_map.policy_action_space # if converting continuous to discrete, otherwise the same
        num_inputs = int(np.prod(state_extractor.obs_shape))
        max_action = action_map.action_space.n if action_map.discrete_actions else 1 # might have problems with discretized actions
    else:
        action_space = environment.action_space
        paction_space = environment.action_space
        num_inputs = environment.observation_space.shape
        max_action = environment.action_space.n if environment.discrete_actions else environment.action_space.high[0]
    option.time_cutoff = args.time_cutoff
    args.option = option
    args.first_obj_dim = state_extractor.first_obj_shape[0]
    args.object_dim = state_extractor.object_shape[0]
    args.post_dim = state_extractor.post_dim
    if args.save_loaded_network:
        policy = LoadedPolicy(load_network=args.load_network)
        action_map.assign_policy_map(policy.map_action, policy.reverse_map_action, policy.exploration_noise)
        option.policy = policy
        if len(args.save_graph) > 0:
            print(args.object)
            graph.nodes[args.object] = OptionNode(args.object, option, action_shape = action_map.mapped_action_shape)
            graph.cfs += dataset_model.cfselectors
            graph.add_edge(OptionEdge(args.object, option_name))
            print(args.save_graph, graph.nodes)
            graph.save_graph(args.save_graph, [args.object], args.environment_model, cuda=args.cuda)
            error
    else:
        if not load_option and not args.change_option:
            args.discrete_actions, args.discretize_acts = args.discretize_acts, args.discrete_actions
            policy = TSPolicy(num_inputs, paction_space, action_space, max_action, **vars(args)) # default args?
            args.discrete_actions, args.discretize_acts = args.discretize_acts, args.discrete_actions
        else:
            set_option = graph.nodes[args.object].option
            policy = set_option.policy
            policy.option = option
        action_map.assign_policy_map(policy.map_action, policy.reverse_map_action, policy.exploration_noise)
        if not load_option:
            option.policy = policy
            policy.option = option
            graph.nodes[args.object] = OptionNode(args.object, option, action_shape = action_map.mapped_action_shape)
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
    
    print("pre buffer", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
    # TODO: only initializes with ReplayBuffer, prioritizedReplayBuffer at the moment, but could extend to vector replay buffer if multithread possible
    if len(args.prioritized_replay) > 0:
        trainbuffer = ParamPriorityReplayBuffer(args.buffer_len, stack_num=1, alpha=args.prioritized_replay[0], beta=args.prioritized_replay[1])
    else:
        trainbuffer = ParamReplayBuffer(args.buffer_len, stack_num=1)

    train_collector = OptionCollector(option.policy, environment, trainbuffer, exploration_noise=True, test=False,
                        option=option, environment_model = environment_model, args = args) # for now, no preprocess function
    if len(args.load_pretrain) > 0:
        pretrain_collector = load_from_pickle(os.path.join(args.load_pretrain, "pretrain_collector.pkl"))
        train_collector.at = pretrain_collector.at
        train_collector.buffer = pretrain_collector.buffer
        train_collector.full_at = pretrain_collector.full_at
        train_collector.full_buffer = pretrain_collector.full_buffer
        if option.policy.learning_algorithm is not None:
            option.policy.learning_algorithm.at = pretrain_collector.hindsight_at
            option.policy.sample_buffer = pretrain_collector.hindsight_buffer
            option.policy.learning_algorithm.replay_buffer = pretrain_collector.hindsight_buffer
    MAXEPISODELEN = 100
    test_collector = OptionCollector(option.policy, test_environment, ParamReplayBuffer(MAXEPISODELEN, 1), option=option, test=True, args=args, environment_model=test_environment_model)

    tensorboard_logger = SummaryWriter(log_dir=os.path.join(args.record_rollouts, "logs") if len(args.record_rollouts) > 0 else "./logs/temp/")

    print("pre train", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
    trained = trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph, tensorboard_logger)
    tensorboard_logger.close()
    if trained and not args.true_environment: # if trained, add control feature to the graph
        graph.cfs += dataset_model.cfselectors
        graph.add_edge(OptionEdge(args.object, option_name))
    if args.train and args.save_interval > 0:
        option.cpu()
        option.save(args.save_dir)
    if len(args.save_graph) > 0:
        print(args.object)
        graph.save_graph(args.save_graph, [args.object], args.environment_model, cuda=args.cuda)