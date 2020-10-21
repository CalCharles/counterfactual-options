import os
from arguments import get_args
from file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from EnvironmentModels.SelfBreakout.breakout_environment_model import BreakoutEnvironmentModel
from Environments.SelfBreakout.breakout_screen import Screen
from ReinforcementLearning.rollouts import RLRollouts, get_RL_shapes
from ReinforcementLearning.train_RL import trainRL, Logger
from ReinforcementLearning.Policy.policy import policy_forms
from ReinforcementLearning.behavior_policy import behavior_forms
from ReinforcementLearning.learning_algorithms import learning_algorithms
from Options.Termination.termination import terminal_forms
from Options.option_graph import OptionGraph, OptionNode, load_graph
from Options.option import Option, PrimitiveOption, option_forms
from Options.Reward.reward import reward_forms
from DistributionalModels.DatasetModels.dataset_model import FactoredDatasetModel
from DistributionalModels.distributional_model import load_factored_model
import torch
import numpy as np

if __name__ == '__main__':
    args = get_args()
    torch.cuda.set_device(args.gpu)
    environment = Screen()
    environment.set_save(0, args.record_rollouts, args.save_recycle, save_raw=args.save_raw)
    environment_model = BreakoutEnvironmentModel(environment)
    dataset_model = load_factored_model(args.dataset_dir)
    if args.cuda:
        dataset_model.cuda()
    # print(dataset_model.observed_outcomes)
    try:
        graph = load_graph(args.graph_dir)
        print("loaded graph from ", args.graph_dir)
    except OSError as e:
        actions = PrimitiveOption(None, None, None, None, environment_model, None, None, "Action", ["Actions"], num_params=environment.num_actions)
        nodes = {'Action': OptionNode('Action', actions, action_shape = (1,), num_params=environment.num_actions)}
        graph = OptionGraph(nodes, dict())
    termination = terminal_forms[args.terminal_type](use_diff=args.use_both==1, use_both=args.use_both==2, name=args.object, min_use=args.min_use, dataset_model=dataset_model, epsilon=args.epsilon_close)
    reward = reward_forms[args.reward_type](use_diff=args.use_both==1, use_both=args.use_both==2, epsilon=args.epsilon_close) # using the same epsilon for now, but that could change
    names = [args.object, dataset_model.option_name]
    load_option = not args.train and args.object in graph.nodes
    print(load_option, args.object)
    if not load_option:
        option = option_forms[args.option_type](None, None, termination, graph.nodes[dataset_model.option_name].option, dataset_model, environment_model, reward, args.object, names, temp_ext=False)
    else:
        option = graph.nodes[args.object].option
    rl_shape_dict = get_RL_shapes(option, environment_model)
    rollouts = RLRollouts(args.buffer_steps, rl_shape_dict)
    print(rl_shape_dict["state"], rl_shape_dict["probs"], rl_shape_dict["param"])
    args.num_inputs, args.num_outputs, args.param_size = rl_shape_dict["state"][0], rl_shape_dict["probs"][0], rl_shape_dict["param"][0]
    if not load_option:
        policy = policy_forms[args.policy_type](**vars(args)) # default args?
    else:
        policy = option.policy
    if args.cuda:
        policy.cuda()
        rollouts = rollouts.cuda()
        option.cuda()
        dataset_model.cuda()
    behavior_policy = behavior_forms[args.behavior_type](args)
    if not load_option:
        option.policy = policy
        option.behavior_policy = behavior_policy
        graph.nodes[args.object] = OptionNode(args.object, option, action_shape = option.action_shape, num_params = len(option.get_possible_parameters()))
    else:
        graph.load_environment_model(environment_model)
    print(load_option, option.policy)
    logger = Logger(args)
    learning_algorithm = learning_algorithms[args.learning_type](args, option)
    if args.set_time_cutoff:
        option.time_cutoff = -1
    done_lengths = trainRL(args, rollouts, logger, environment, environment_model, option, learning_algorithm, names, graph)
    done_lengths = np.array(done_lengths)
    time_cutoff = 100
    if len(done_lengths) > 0:
        time_cutoff = np.round_(np.quantile(done_lengths, .9))

    if args.train and args.save_interval > 0:
        option.save(args.save_dir)
    if args.set_time_cutoff:
        option.time_cutoff = time_cutoff
    if len(args.save_graph) > 0:
        print(args.object, graph.nodes[args.object].option.time_cutoff)
        graph.save_graph(args.save_graph, [args.object])
    print (time_cutoff)
